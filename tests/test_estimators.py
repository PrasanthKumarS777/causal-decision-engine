import numpy as np
import pytest
from src.data.generator import CustomerDataGenerator, DatasetConfig
from src.policy.bandit import ThompsonSamplingBandit, BanditConfig


@pytest.fixture(scope="module")
def small_df():
    return CustomerDataGenerator(DatasetConfig(n_samples=500)).generate()


class TestDataGenerator:
    def test_shape(self, small_df):
        assert len(small_df) == 500

    def test_treatment_values(self, small_df):
        assert small_df["treatment"].isin([0, 1, 2]).all()

    def test_outcome_binary(self, small_df):
        assert small_df["customer_retained"].isin([0, 1]).all()

    def test_no_nulls(self, small_df):
        assert small_df.isnull().sum().sum() == 0

    def test_clv_positive(self, small_df):
        assert (small_df["clv"] > 0).all()


class TestThompsonSamplingBandit:
    def test_arm_selection_valid(self):
        b = ThompsonSamplingBandit()
        assert b.select_arm() in [0, 1, 2]

    def test_arm_selection_with_cate(self):
        b    = ThompsonSamplingBandit()
        cate = np.array([0.0, 0.08, 0.18])
        assert b.select_arm(cate=cate) in [0, 1, 2]

    def test_update_success(self):
        b = ThompsonSamplingBandit()
        b.update(arm=1, reward=1)
        assert b.alpha[1] == 2.0
        assert b.beta_[1] == 1.0

    def test_update_failure(self):
        b = ThompsonSamplingBandit()
        b.update(arm=0, reward=0)
        assert b.alpha[0] == 1.0
        assert b.beta_[0] == 2.0

    def test_invalid_reward_raises(self):
        b = ThompsonSamplingBandit()
        with pytest.raises(ValueError):
            b.update(arm=0, reward=5)

    def test_optimal_policy_shape(self):
        b   = ThompsonSamplingBandit()
        mat = np.array([[0.0, 0.08, 0.18], [0.0, 0.15, 0.10]])
        pol = b.optimal_policy(mat)
        assert pol.shape == (2,)

    def test_optimal_policy_picks_highest_cate(self):
        b   = ThompsonSamplingBandit(BanditConfig(arm_costs=[0.0, 0.0, 0.0]))
        mat = np.array([[0.0, 0.08, 0.18]])
        assert b.optimal_policy(mat)[0] == 2

    def test_state_keys(self):
        b     = ThompsonSamplingBandit()
        state = b.state()
        for key in ["alpha", "beta", "expected_rewards", "recommended_arm", "total_updates"]:
            assert key in state

    def test_expected_rewards_range(self):
        b   = ThompsonSamplingBandit()
        exp = b.expected_rewards()
        assert all(0 < r < 1 for r in exp)
