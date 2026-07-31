import unittest
from pathlib import Path
from mal2026.solar_prompt_search_v4 import SearchConfigV4,base,request_specs,train_splits
class SolarPromptSearchV4Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.config=SearchConfigV4.from_json(Path("configs/solar_prompt_search.v4.json"))
    def test_oof_base_and_shapes(self):
        row=train_splits(self.config)["discovery"][0];prediction=base(self.config,row.identifier)
        self.assertEqual(set(prediction),{"content","organization","expression"})
        self.assertEqual(len(request_specs(self.config,"residual8_joint_delta",row)),1)
        self.assertEqual(len(request_specs(self.config,"residual8_axis_delta",row)),3)
if __name__=="__main__":unittest.main()
