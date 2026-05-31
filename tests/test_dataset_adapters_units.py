from __future__ import annotations

import unittest


class DatasetAdaptersUnitTests(unittest.TestCase):
    def test_select_latest_event_picks_max_year_round(self) -> None:
        import pandas as pd

        from src.dataset_adapters import _select_latest_event

        df = pd.DataFrame(
            {
                "year": [2022, 2023, 2023, None],
                "round": [1, 2, 3, 99],
            }
        )
        year, rnd = _select_latest_event(df)
        self.assertEqual(year, 2023)
        self.assertEqual(rnd, 3)

    def test_select_latest_event_raises_when_empty(self) -> None:
        import pandas as pd

        from src.dataset_adapters import _select_latest_event

        with self.assertRaises(ValueError):
            _select_latest_event(pd.DataFrame({"year": [], "round": []}))

    def test_select_best_driver_prefers_most_laps_then_best_finish(self) -> None:
        import pandas as pd

        from src.dataset_adapters import _select_best_driver

        df = pd.DataFrame(
            {
                "code": ["AAA", "AAA", "BBB", "BBB", "BBB", "CCC", "CCC"],
                "lap": [1, 2, 1, 2, 3, 1, 2],
                # Lower is better. BBB has more laps anyway; for tie-break test below we also
                # include CCC.
                "position_y": [5, 5, 8, 8, 8, 1, 1],
            }
        )
        self.assertEqual(_select_best_driver(df), "BBB")

        # Tie on lap count: choose best (min) finish position.
        df2 = pd.DataFrame(
            {
                "code": ["AAA", "AAA", "CCC", "CCC"],
                "lap": [1, 2, 1, 2],
                "position_y": [5, 5, 2, 2],
            }
        )
        self.assertEqual(_select_best_driver(df2), "CCC")

    def test_select_best_driver_tiebreaks_by_code(self) -> None:
        import pandas as pd

        from src.dataset_adapters import _select_best_driver

        df = pd.DataFrame(
            {
                "code": ["ZZZ", "ZZZ", "AAA", "AAA"],
                "lap": [1, 2, 1, 2],
                "position_y": [3, 3, 3, 3],
            }
        )
        # Same laps/position, code ascending.
        self.assertEqual(_select_best_driver(df), "AAA")

    def test_select_best_driver_raises_when_no_codes(self) -> None:
        import pandas as pd

        from src.dataset_adapters import _select_best_driver

        with self.assertRaises(ValueError):
            _select_best_driver(pd.DataFrame({"code": [None, None], "lap": [1, 2], "position_y": [1, 2]}))

    def test_select_first_stint_decision_lap_detects_tyre_age_reset(self) -> None:
        import pandas as pd

        from src.dataset_adapters import _select_first_stint_decision_lap

        df = pd.DataFrame(
            {
                "lap": [1, 2, 3, 4, 5],
                "TyreLife": [1, 2, 3, 1, 2],
                "Compound": ["MEDIUM", "MEDIUM", "MEDIUM", "MEDIUM", "MEDIUM"],
            }
        )
        # Reset happens between lap 3 -> 4; decision lap should be 3.
        self.assertEqual(_select_first_stint_decision_lap(df), 3)

    def test_select_first_stint_decision_lap_detects_compound_change(self) -> None:
        import pandas as pd

        from src.dataset_adapters import _select_first_stint_decision_lap

        df = pd.DataFrame(
            {
                "lap": [1, 2, 3, 4],
                "TyreLife": [1, 2, 3, 4],
                "Compound": ["SOFT", "SOFT", "HARD", "HARD"],
            }
        )
        # Change between lap 2 -> 3; decision lap should be 2.
        self.assertEqual(_select_first_stint_decision_lap(df), 2)

    def test_select_first_stint_decision_lap_defaults_to_max_lap(self) -> None:
        import pandas as pd

        from src.dataset_adapters import _select_first_stint_decision_lap

        df = pd.DataFrame(
            {
                "lap": [1, 2, 3],
                "TyreLife": [1, 2, 3],
                "Compound": ["SOFT", "SOFT", "SOFT"],
            }
        )
        self.assertEqual(_select_first_stint_decision_lap(df), 3)


if __name__ == "__main__":
    unittest.main()
