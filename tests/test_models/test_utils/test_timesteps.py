from unittest import TestCase

from diffengine.models.utils import WuerstchenRandomTimeSteps


class TestWuerstchenRandomTimeSteps(TestCase):

    def test_init(self):
        _ = WuerstchenRandomTimeSteps()

    def test_forward(self):
        module = WuerstchenRandomTimeSteps()
        batch_size = 2
        timesteps = module(batch_size, "cpu")
        assert timesteps.shape == (2,)
