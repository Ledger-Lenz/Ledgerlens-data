// Soroban contract: M-of-N multi-signature governance for RISK_SCORE_FLAG_THRESHOLD.
//
// Deploy with M and N set at construction time (default M=2, N=3).
// Proposals expire after PROPOSAL_TTL_LEDGERS (≈7 days at ~5 s/ledger).
//
// Entry points
// ─────────────
// propose_threshold_change(proposer: Address, new_threshold: u32) → u64 (proposal_id)
// approve_threshold_change(approver: Address, proposal_id: u64) → bool (applied?)
// current_threshold() → u32
// get_proposal(proposal_id: u64) → Proposal

#![no_std]
use soroban_sdk::{
    contract, contractimpl, contracttype, symbol_short,
    Address, Env, Map, Vec,
};

const PROPOSAL_TTL_LEDGERS: u32 = 120_960; // 7 days @ ~5 s per ledger

#[contracttype]
#[derive(Clone)]
pub struct Proposal {
    pub new_threshold: u32,
    pub proposer: Address,
    pub approvals: Vec<Address>,
    pub expiry_ledger: u32,
    pub applied: bool,
}

#[contract]
pub struct ThresholdGovernanceContract;

#[contractimpl]
impl ThresholdGovernanceContract {
    /// One-time initialisation: set keyholders (N addresses) and quorum (M).
    pub fn init(env: Env, keyholders: Vec<Address>, quorum: u32, initial_threshold: u32) {
        if env.storage().instance().has(&symbol_short!("init")) {
            panic!("already initialised");
        }
        assert!(
            quorum >= 1 && (quorum as usize) <= keyholders.len(),
            "quorum must be between 1 and N"
        );
        env.storage().instance().set(&symbol_short!("keys"), &keyholders);
        env.storage().instance().set(&symbol_short!("quorum"), &quorum);
        env.storage().instance().set(&symbol_short!("thresh"), &initial_threshold);
        env.storage().instance().set(&symbol_short!("next_id"), &0u64);
        env.storage().instance().set(&symbol_short!("init"), &true);
    }

    /// Start a governance vote to change the threshold.
    /// Returns the new proposal_id.
    pub fn propose_threshold_change(env: Env, proposer: Address, new_threshold: u32) -> u64 {
        proposer.require_auth();
        Self::require_keyholder(&env, &proposer);
        assert!(new_threshold <= 100, "threshold must be 0–100");

        let proposal_id: u64 = env.storage().instance().get(&symbol_short!("next_id")).unwrap();
        let proposal = Proposal {
            new_threshold,
            proposer: proposer.clone(),
            approvals: Vec::from_array(&env, [proposer]),
            expiry_ledger: env.ledger().sequence() + PROPOSAL_TTL_LEDGERS,
            applied: false,
        };

        let mut proposals: Map<u64, Proposal> = env
            .storage()
            .instance()
            .get(&symbol_short!("props"))
            .unwrap_or(Map::new(&env));
        proposals.set(proposal_id, proposal);
        env.storage().instance().set(&symbol_short!("props"), &proposals);
        env.storage().instance().set(&symbol_short!("next_id"), &(proposal_id + 1));

        env.events().publish(
            (symbol_short!("proposed"), proposal_id),
            new_threshold,
        );
        proposal_id
    }

    /// Cast an approval for an open proposal.
    /// Returns true if quorum was reached and the threshold was applied.
    pub fn approve_threshold_change(env: Env, approver: Address, proposal_id: u64) -> bool {
        approver.require_auth();
        Self::require_keyholder(&env, &approver);

        let mut proposals: Map<u64, Proposal> = env
            .storage()
            .instance()
            .get(&symbol_short!("props"))
            .unwrap();
        let mut p = proposals.get(proposal_id).expect("proposal not found");

        assert!(!p.applied, "proposal already applied");
        assert!(
            env.ledger().sequence() <= p.expiry_ledger,
            "proposal expired"
        );
        assert!(!p.approvals.contains(&approver), "already approved");

        p.approvals.push_back(approver);

        let quorum: u32 = env.storage().instance().get(&symbol_short!("quorum")).unwrap();
        let applied = (p.approvals.len() as u32) >= quorum;

        if applied {
            p.applied = true;
            env.storage().instance().set(&symbol_short!("thresh"), &p.new_threshold);
            env.events().publish(
                (symbol_short!("t_changed"), proposal_id),
                p.new_threshold,
            );
        }

        proposals.set(proposal_id, p);
        env.storage().instance().set(&symbol_short!("props"), &proposals);
        applied
    }

    pub fn current_threshold(env: Env) -> u32 {
        env.storage().instance().get(&symbol_short!("thresh")).unwrap()
    }

    pub fn get_proposal(env: Env, proposal_id: u64) -> Proposal {
        let proposals: Map<u64, Proposal> = env
            .storage()
            .instance()
            .get(&symbol_short!("props"))
            .unwrap();
        proposals.get(proposal_id).expect("proposal not found")
    }

    fn require_keyholder(env: &Env, addr: &Address) {
        let keys: Vec<Address> = env.storage().instance().get(&symbol_short!("keys")).unwrap();
        assert!(keys.contains(addr), "not a keyholder");
    }
}
