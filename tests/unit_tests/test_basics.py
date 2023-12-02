import pytest
from anime2sd.basics import parse_anime_info


@pytest.mark.parametrize(
    "filename, expected",
    [
        (
            "[SubsPlease] 16bit Sensation - Another Layer - 07 (1080p) [771BDD0C].mkv",
            ("16bit Sensation - Another Layer", 7),
        ),
        (
            "[HorribleSubs] Toaru Kagaku no Railgun T - 25 [1080p].mkv",
            ("Toaru Kagaku no Railgun T", 25),
        ),
        ("[Hayaisubs] Yama no Susume 2 - 18 [720p].mkv", ("Yama no Susume 2", 18)),
        # Add more test cases as needed
        (
            "[RandomGroup] Anime Title - Extra Info - 10 [720p].mkv",
            ("Anime Title - Extra Info", 10),
        ),
        (
            "Yama no Susume (Saison 2) 16 vostfr [720p]",
            ("Yama no Susume (Saison 2) 16 vostfr", None),
        ),
        (
            "[Ohys-Raws] Toaru Kagaku no Railgun T - SP2 (BD 1280x720 x264 AAC).mp4",
            ("Toaru Kagaku no Railgun T", None),
        ),
        (
            "[EA]Toaru_Kagaku_no_Railgun_T_24_[1920x1080][Hi10p][373BAEBF].mkv",
            ("Toaru_Kagaku_no_Railgun_T_24_", None),
        ),
        ("Only Title.mkv", ("Only Title", None)),
        ("[Group] Only Title - No Episode.mkv", ("Only Title", None)),
    ],
)
def test_parse_anime_info(filename, expected):
    assert parse_anime_info(filename) == expected
