"""
Integration tests for training module MCP tools

Tests training tools using FastMCP integration with mocked Garmin API responses.
"""
import pytest
from mcp.server.fastmcp import FastMCP

import json

from garminconnect import (
    GarminConnectAuthenticationError,
    GarminConnectConnectionError,
    GarminConnectTooManyRequestsError,
)

from garmin_mcp import training
from tests.fixtures.garmin_responses import (
    MOCK_PROGRESS_SUMMARY,
    MOCK_HRV_DATA,
    MOCK_TRAINING_STATUS,
    MOCK_LACTATE_THRESHOLD,
    MOCK_LACTATE_THRESHOLD_RANGE,
    MOCK_CYCLING_FTP,
    MOCK_ENDURANCE_SCORE,
    MOCK_ACTIVITY_TYPES,
)


@pytest.fixture
def app_with_training(mock_garmin_client):
    """Create FastMCP app with training tools registered"""
    training.configure(mock_garmin_client)
    app = FastMCP("Test Training")
    app = training.register_tools(app)
    return app


@pytest.mark.asyncio
async def test_get_progress_summary_between_dates_tool(app_with_training, mock_garmin_client):
    """Test get_progress_summary_between_dates tool"""
    # Setup mock
    mock_garmin_client.get_progress_summary_between_dates.return_value = MOCK_PROGRESS_SUMMARY

    # Call tool
    result = await app_with_training.call_tool(
        "get_progress_summary_between_dates",
        {
            "start_date": "2024-01-08",
            "end_date": "2024-01-15",
            "metric": "duration"
        }
    )

    # Verify
    assert result is not None
    mock_garmin_client.get_progress_summary_between_dates.assert_called_once_with(
        "2024-01-08", "2024-01-15", "duration"
    )


@pytest.mark.asyncio
async def test_get_hill_score_tool(app_with_training, mock_garmin_client):
    """Test get_hill_score tool"""
    # Setup mock
    hill_score = {
        "hillScore": 75,
        "dateRange": {"start": "2024-01-08", "end": "2024-01-15"}
    }
    mock_garmin_client.get_hill_score.return_value = hill_score

    # Call tool
    result = await app_with_training.call_tool(
        "get_hill_score",
        {"start_date": "2024-01-08", "end_date": "2024-01-15"}
    )

    # Verify
    assert result is not None
    mock_garmin_client.get_hill_score.assert_called_once_with("2024-01-08", "2024-01-15")


@pytest.mark.asyncio
async def test_get_endurance_score_tool(app_with_training, mock_garmin_client):
    """Test get_endurance_score tool with realistic API response"""
    # Setup mocks
    mock_garmin_client.get_endurance_score.return_value = MOCK_ENDURANCE_SCORE
    mock_garmin_client.get_activity_types.return_value = MOCK_ACTIVITY_TYPES

    # Reset the activity type cache to ensure fresh lookup
    training._activity_type_cache = None

    # Call tool
    result = await app_with_training.call_tool(
        "get_endurance_score",
        {"start_date": "2024-01-08", "end_date": "2024-01-15"}
    )

    # Verify API was called correctly
    assert result is not None
    mock_garmin_client.get_endurance_score.assert_called_once_with("2024-01-08", "2024-01-15")

    data = json.loads(result[0][0].text)

    # Check period summary
    assert data["period_avg_score"] == 5631
    assert data["period_max_score"] == 5740

    # Check current score
    assert data["current_score"] == 5712
    assert data["current_date"] == "2024-01-15"
    assert data["classification"] == "intermediate"
    assert data["classification_id"] == 2

    # Check thresholds
    assert "thresholds" in data
    assert data["thresholds"]["trained"] == 5800
    assert data["thresholds"]["well_trained"] == 6500

    # Check contributors have activity type names
    assert "contributors" in data
    contributors = data["contributors"]
    assert len(contributors) == 4

    # Find the hiking contributor
    hiking_contributor = next(
        (c for c in contributors if c.get("activity_type") == "hiking"), None
    )
    assert hiking_contributor is not None
    assert hiking_contributor["contribution_percent"] == 5.49
    assert hiking_contributor["activity_type_id"] == 3

    # Find the yoga contributor
    yoga_contributor = next(
        (c for c in contributors if c.get("activity_type") == "yoga"), None
    )
    assert yoga_contributor is not None
    assert yoga_contributor["contribution_percent"] == 3.13

    # Check weekly breakdown exists
    assert "weekly_breakdown" in data
    assert len(data["weekly_breakdown"]) == 1
    week = data["weekly_breakdown"][0]
    assert week["week_start"] == "2024-01-08"
    assert week["avg_score"] == 5548
    assert week["max_score"] == 5561


@pytest.mark.asyncio
async def test_get_running_tolerance_tool(app_with_training, mock_garmin_client):
    """Running tolerance keeps dated values and does not fill gaps with zero."""
    mock_garmin_client.get_running_tolerance.return_value = [
        {
            "userProfilePK": 1,
            "calendarDate": "2026-03-15",
            "totalImpactLoad": 53800,
            "totalDistance": 49615.0,
            "tolerance": 60914,
            "startOfWeek": "2026-03-11",
            "endOfWeek": "2026-03-15",
            "weekIndex": 1888,
        },
        {
            "userProfilePK": 1,
            "calendarDate": "2026-03-18",
            "totalImpactLoad": 14610,
            "totalDistance": 11036.0,
            "tolerance": 0,
            "startOfWeek": "2026-03-16",
            "endOfWeek": "2026-03-18",
            "weekIndex": 1889,
        },
    ]

    result = await app_with_training.call_tool(
        "get_running_tolerance",
        {
            "start_date": "2026-03-11",
            "end_date": "2026-03-18",
            "aggregation": "weekly",
        },
    )
    data = json.loads(result[0][0].text)
    assert data["count"] == 2
    assert data["no_data"] is False
    assert data["coverage"]["filled_with_zero"] is False
    assert data["score_does_not_authorize"] is True
    assert data["observations"][0]["date"] == "2026-03-15"
    assert data["observations"][0]["tolerance"] == 60914
    assert data["observations"][1]["tolerance"] == 0
    mock_garmin_client.get_running_tolerance.assert_called_once_with(
        "2026-03-11", "2026-03-18", "weekly"
    )


@pytest.mark.asyncio
async def test_get_running_tolerance_empty_is_not_zero(app_with_training, mock_garmin_client):
    mock_garmin_client.get_running_tolerance.return_value = []
    result = await app_with_training.call_tool(
        "get_running_tolerance",
        {"start_date": "2026-03-11", "end_date": "2026-03-18"},
    )
    data = json.loads(result[0][0].text)
    assert data["count"] == 0
    assert data["no_data"] is True
    assert data["observations"] == []
    assert data["coverage"]["filled_with_zero"] is False
    assert 0 not in [row.get("tolerance") for row in data["observations"]]


@pytest.mark.asyncio
async def test_get_running_tolerance_classifies_auth_failure(app_with_training, mock_garmin_client):
    mock_garmin_client.get_running_tolerance.side_effect = (
        GarminConnectAuthenticationError("expired")
    )
    result = await app_with_training.call_tool(
        "get_running_tolerance",
        {"start_date": "2026-03-11", "end_date": "2026-03-18"},
    )
    data = json.loads(result[0][0].text)
    assert data["no_data"] is False
    assert data["failure"]["kind"] == "authentication"
    assert data["coverage"]["failures"] == ["authentication"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("start_date", "end_date"),
    [
        ("2026-99-99", "2026-03-18"),
        ("2026-03-19", "2026-03-18"),
    ],
)
async def test_get_running_tolerance_rejects_invalid_window(
    app_with_training, mock_garmin_client, start_date, end_date
):
    result = await app_with_training.call_tool(
        "get_running_tolerance",
        {"start_date": start_date, "end_date": end_date},
    )
    data = json.loads(result[0][0].text)
    assert data["failure"]["kind"] == "validation"
    mock_garmin_client.get_running_tolerance.assert_not_called()


@pytest.mark.asyncio
async def test_get_training_effect_tool(app_with_training, mock_garmin_client):
    """Test get_training_effect tool"""
    # Setup mock - get_training_effect uses get_activity internally
    activity_data = {
        "summaryDTO": {
            "trainingEffect": 3.5,
            "anaerobicTrainingEffect": 2.0,
            "trainingEffectLabel": "Highly Improving",
            "activityTrainingLoad": 150,
            "recoveryTime": 720,  # 12 hours in minutes
            "performanceCondition": 95,
        }
    }
    mock_garmin_client.get_activity.return_value = activity_data

    # Call tool
    result = await app_with_training.call_tool(
        "get_training_effect",
        {"activity_id": 12345678901}
    )

    # Verify
    assert result is not None
    mock_garmin_client.get_activity.assert_called_once_with(12345678901)


def _hrv_payload(last_night_avg, *, weekly_avg=45, calendar_date="2024-01-15"):
    """Synthetic Garmin HRV payload. lastNightAvg is the observed overnight field.

    last_night_avg=None omits lastNightAvg (absence). Never includes lastNight —
    that name is not the overnight average in Garmin's hrvSummary.
    """
    payload = json.loads(json.dumps(MOCK_HRV_DATA))
    summary = payload["hrvSummary"]
    summary["calendarDate"] = calendar_date
    summary["weeklyAvg"] = weekly_avg
    summary.pop("lastNight", None)
    if last_night_avg is None:
        summary.pop("lastNightAvg", None)
    else:
        summary["lastNightAvg"] = last_night_avg
    return payload


def _tool_json(result):
    return json.loads(result[0][0].text)


@pytest.mark.asyncio
async def test_get_hrv_data_tool(app_with_training, mock_garmin_client):
    """Test get_hrv_data tool"""
    mock_garmin_client.get_hrv_data.return_value = MOCK_HRV_DATA

    result = await app_with_training.call_tool(
        "get_hrv_data",
        {"date": "2024-01-15"}
    )

    data = _tool_json(result)
    assert data["last_night_avg_hrv_ms"] == 48
    mock_garmin_client.get_hrv_data.assert_called_once_with("2024-01-15")


@pytest.mark.asyncio
async def test_hrv_daily_and_trend_agree_on_last_night_avg(app_with_training, mock_garmin_client):
    """Same lastNightAvg observation must match between daily and trend (T08)."""
    mock_garmin_client.get_hrv_data.side_effect = lambda date: _hrv_payload(48, calendar_date=date)

    daily = _tool_json(await app_with_training.call_tool("get_hrv_data", {"date": "2024-01-15"}))
    trend = _tool_json(await app_with_training.call_tool(
        "get_hrv_trend",
        {"start_date": "2024-01-15", "end_date": "2024-01-15"},
    ))

    assert daily["last_night_avg_hrv_ms"] == 48
    assert trend["trend"][0]["last_night_avg_hrv_ms"] == 48
    assert trend["period_avg_hrv_ms"] == 48
    assert trend["hrv_sample_count"] == 1


@pytest.mark.asyncio
async def test_hrv_zero_is_not_missing(app_with_training, mock_garmin_client):
    """Overnight HRV of 0 is a valid sample, distinct from an omitted field (T03)."""
    mock_garmin_client.get_hrv_data.side_effect = lambda date: _hrv_payload(0, calendar_date=date)

    daily = _tool_json(await app_with_training.call_tool("get_hrv_data", {"date": "2024-01-15"}))
    trend = _tool_json(await app_with_training.call_tool(
        "get_hrv_trend",
        {"start_date": "2024-01-15", "end_date": "2024-01-15"},
    ))

    assert daily["last_night_avg_hrv_ms"] == 0
    assert trend["trend"][0]["last_night_avg_hrv_ms"] == 0
    assert trend["hrv_sample_count"] == 1
    assert trend["period_avg_hrv_ms"] == 0


@pytest.mark.asyncio
async def test_hrv_trend_ignores_unproven_last_night_alias(app_with_training, mock_garmin_client):
    """lastNight is not a proven alias of lastNightAvg — do not copy it into the average."""
    payload = _hrv_payload(None)
    payload["hrvSummary"]["lastNight"] = 99
    mock_garmin_client.get_hrv_data.return_value = payload

    daily = _tool_json(await app_with_training.call_tool("get_hrv_data", {"date": "2024-01-15"}))
    trend = _tool_json(await app_with_training.call_tool(
        "get_hrv_trend",
        {"start_date": "2024-01-15", "end_date": "2024-01-15"},
    ))

    assert "last_night_avg_hrv_ms" not in daily
    assert "last_night_avg_hrv_ms" not in trend["trend"][0]
    assert trend.get("hrv_sample_count") == 0
    assert trend.get("period_avg_hrv_ms") is None


@pytest.mark.asyncio
async def test_hrv_period_average_skips_missing_nights(app_with_training, mock_garmin_client):
    """Period average uses only nights with lastNightAvg; missing days are not zeros."""

    def by_date(date):
        values = {
            "2024-01-15": 48,
            "2024-01-16": None,
            "2024-01-17": 0,
        }
        return _hrv_payload(values[date], calendar_date=date)

    mock_garmin_client.get_hrv_data.side_effect = by_date

    trend = _tool_json(await app_with_training.call_tool(
        "get_hrv_trend",
        {"start_date": "2024-01-15", "end_date": "2024-01-17"},
    ))

    by_day = {row["date"]: row for row in trend["trend"]}
    assert by_day["2024-01-15"]["last_night_avg_hrv_ms"] == 48
    assert "last_night_avg_hrv_ms" not in by_day["2024-01-16"]
    assert by_day["2024-01-17"]["last_night_avg_hrv_ms"] == 0
    assert trend["hrv_sample_count"] == 2
    assert trend["period_avg_hrv_ms"] == 24.0


@pytest.mark.asyncio
async def test_get_fitnessage_data_tool(app_with_training, mock_garmin_client):
    """Test get_fitnessage_data tool"""
    # Setup mock
    fitness_age = {
        "fitnessAge": 25,
        "chronologicalAge": 30,
        "vo2Max": 52.5,
        "date": "2024-01-15"
    }
    mock_garmin_client.get_fitnessage_data.return_value = fitness_age

    # Call tool
    result = await app_with_training.call_tool(
        "get_fitnessage_data",
        {"date": "2024-01-15"}
    )

    # Verify
    assert result is not None
    mock_garmin_client.get_fitnessage_data.assert_called_once_with("2024-01-15")


@pytest.mark.asyncio
async def test_get_cycling_ftp_tool(app_with_training, mock_garmin_client):
    """Test get_cycling_ftp tool returns latest FTP data"""
    mock_garmin_client.get_cycling_ftp.return_value = MOCK_CYCLING_FTP

    result = await app_with_training.call_tool("get_cycling_ftp", {})

    assert result is not None
    mock_garmin_client.get_cycling_ftp.assert_called_once_with()

    data = json.loads(result[0][0].text)
    assert data["sport"] == "CYCLING"
    assert data["functional_threshold_power_watts"] == 294
    assert data["calendar_date"] == "2024-03-15T10:30:00.000"
    assert data["is_stale"] is False
    assert data["biometric_source_type"] == "CHANGE_LOG"


@pytest.mark.asyncio
async def test_request_reload_tool(app_with_training, mock_garmin_client):
    """Test request_reload tool"""
    # Setup mock
    reload_response = {"status": "success", "message": "Data reload requested"}
    mock_garmin_client.request_reload.return_value = reload_response

    # Call tool
    result = await app_with_training.call_tool(
        "request_reload",
        {"date": "2024-01-15"}
    )

    # Verify
    assert result is not None
    mock_garmin_client.request_reload.assert_called_once_with("2024-01-15")


@pytest.mark.asyncio
async def test_get_training_status_tool(app_with_training, mock_garmin_client):
    """Test get_training_status tool returns training status"""
    # Setup mock
    mock_garmin_client.get_training_status.return_value = MOCK_TRAINING_STATUS

    # Call tool
    result = await app_with_training.call_tool(
        "get_training_status",
        {"date": "2024-01-15"}
    )

    # Verify
    assert result is not None
    mock_garmin_client.get_training_status.assert_called_once_with("2024-01-15")


@pytest.mark.asyncio
async def test_get_vo2max_trend_falls_back_to_profile(
    app_with_training, mock_garmin_client
):
    """Test current profile estimate is separate from unavailable history"""
    mock_garmin_client.garmin_connect_metrics_url = (
        "/metrics-service/metrics/maxmet/daily"
    )
    mock_garmin_client.connectapi.return_value = []
    mock_garmin_client.get_training_status.return_value = {}
    mock_garmin_client.get_user_profile.return_value = {
        "userData": {"vo2MaxRunning": 28.0}
    }

    result = await app_with_training.call_tool(
        "get_vo2max_trend", {"start_date": "2024-01-14", "end_date": "2024-01-15"}
    )

    data = json.loads(result[0][0].text)
    assert data["data_points"] == 0
    assert data["first_vo2_max"] is None
    assert data["latest_vo2_max"] is None
    assert data["change"] is None
    assert data["trend"] == []
    assert data["current_vo2_max_estimate"] == {
        "vo2_max": 28.0,
        "sport": "running",
        "source": "get_user_profile",
    }
    assert "Historical VO2 max values were not available" in data["note"]
    assert data["coverage"] == {
        "requested": 2,
        "available": 0,
        "missing": 2,
        "failed": 0,
        "stale": 0,
    }
    assert mock_garmin_client.get_training_status.call_count == 2
    mock_garmin_client.connectapi.assert_called_once_with(
        "/metrics-service/metrics/maxmet/daily/2024-01-14/2024-01-15"
    )
    mock_garmin_client.get_max_metrics.assert_not_called()
    mock_garmin_client.get_fitnessage_data.assert_not_called()
    mock_garmin_client.get_user_profile.assert_called_once_with()


@pytest.mark.asyncio
async def test_get_vo2max_trend_uses_daily_metrics(
    app_with_training, mock_garmin_client
):
    """Test range metrics take precedence without daily training-status calls"""
    mock_garmin_client.garmin_connect_metrics_url = (
        "/metrics-service/metrics/maxmet/daily"
    )
    mock_garmin_client.get_training_status.return_value = {
        "mostRecentVO2Max": {"generic": {"vo2MaxValue": 99.0}}
    }
    mock_garmin_client.connectapi.return_value = [
        {
            "generic": {
                "calendarDate": "2024-01-14",
                "vo2MaxValue": 27.5,
            }
        },
        {
            "generic": {
                "calendarDate": "2024-01-15",
                "vo2MaxValue": 28.0,
            }
        },
    ]

    result = await app_with_training.call_tool(
        "get_vo2max_trend", {"start_date": "2024-01-14", "end_date": "2024-01-15"}
    )

    data = json.loads(result[0][0].text)
    assert data["latest_vo2_max"] == 28.0
    assert data["change"] == 0.5
    assert data["sport"] == "running"
    assert data["trend"] == [
        {
            "date": "2024-01-14",
            "vo2_max": 27.5,
            "source": "get_max_metrics",
        },
        {
            "date": "2024-01-15",
            "vo2_max": 28.0,
            "source": "get_max_metrics",
        },
    ]
    mock_garmin_client.connectapi.assert_called_once_with(
        "/metrics-service/metrics/maxmet/daily/2024-01-14/2024-01-15"
    )
    mock_garmin_client.get_training_status.assert_not_called()
    mock_garmin_client.get_max_metrics.assert_not_called()
    mock_garmin_client.get_fitnessage_data.assert_not_called()
    mock_garmin_client.get_user_profile.assert_not_called()


@pytest.mark.asyncio
async def test_get_vo2max_trend_prefers_training_status(
    app_with_training, mock_garmin_client
):
    """Test common historical data avoids extra fallback requests"""
    mock_garmin_client.get_training_status.side_effect = [
        {"mostRecentVO2Max": {"generic": {"vo2MaxValue": 48.0}}},
        {"mostRecentVO2Max": {"generic": {"vo2MaxValue": 48.5}}},
    ]

    result = await app_with_training.call_tool(
        "get_vo2max_trend", {"start_date": "2024-01-14", "end_date": "2024-01-15"}
    )

    data = json.loads(result[0][0].text)
    assert data["latest_vo2_max"] == 48.5
    assert data["sport"] == "running"
    assert all(point["source"] == "get_training_status" for point in data["trend"])
    assert mock_garmin_client.get_training_status.call_count == 2
    mock_garmin_client.get_max_metrics.assert_not_called()
    mock_garmin_client.get_fitnessage_data.assert_not_called()


@pytest.mark.asyncio
async def test_get_vo2max_trend_preserves_selected_cycling_from_mixed_payloads(
    app_with_training, mock_garmin_client
):
    """Test mixed payloads do not hide cycling after it becomes available first"""
    mock_garmin_client.garmin_connect_metrics_url = (
        "/metrics-service/metrics/maxmet/daily"
    )
    mock_garmin_client.connectapi.return_value = []
    mock_garmin_client.get_training_status.side_effect = [
        {"mostRecentVO2Max": {"cycling": {"vo2MaxValue": 55.0}}},
        {
            "mostRecentVO2Max": {
                "generic": {"vo2MaxValue": 48.0},
                "cycling": {"vo2MaxValue": 56.0},
            }
        },
        {
            "mostRecentVO2Max": {
                "generic": {"vo2MaxValue": 48.5},
                "cycling": {"vo2MaxValue": 57.0},
            }
        },
    ]

    result = await app_with_training.call_tool(
        "get_vo2max_trend", {"start_date": "2024-01-13", "end_date": "2024-01-15"}
    )

    data = json.loads(result[0][0].text)
    assert data["sport"] == "cycling"
    assert data["change"] == 2.0
    assert data["trend"] == [
        {
            "date": "2024-01-13",
            "vo2_max": 55.0,
            "source": "get_training_status",
        },
        {
            "date": "2024-01-14",
            "vo2_max": 56.0,
            "source": "get_training_status",
        },
        {
            "date": "2024-01-15",
            "vo2_max": 57.0,
            "source": "get_training_status",
        },
    ]
    mock_garmin_client.get_max_metrics.assert_not_called()


@pytest.mark.asyncio
async def test_get_vo2max_trend_selects_sport_with_most_history(
    app_with_training, mock_garmin_client
):
    """Test an isolated oldest point does not decide the sport for the interval"""
    mock_garmin_client.garmin_connect_metrics_url = (
        "/metrics-service/metrics/maxmet/daily"
    )
    mock_garmin_client.connectapi.return_value = []
    mock_garmin_client.get_training_status.side_effect = [
        {"mostRecentVO2Max": {"cycling": {"vo2MaxValue": 55.0}}},
        {"mostRecentVO2Max": {"generic": {"vo2MaxValue": 48.0}}},
        {"mostRecentVO2Max": {"generic": {"vo2MaxValue": 48.5}}},
    ]

    result = await app_with_training.call_tool(
        "get_vo2max_trend", {"start_date": "2024-01-13", "end_date": "2024-01-15"}
    )

    data = json.loads(result[0][0].text)
    assert data["sport"] == "running"
    assert data["data_points"] == 2
    assert [point["vo2_max"] for point in data["trend"]] == [48.0, 48.5]


@pytest.mark.asyncio
async def test_get_vo2max_trend_uses_cycling_range_metrics(
    app_with_training, mock_garmin_client
):
    """Test cycling-only max metrics retain their sport and date"""
    mock_garmin_client.garmin_connect_metrics_url = (
        "/metrics-service/metrics/maxmet/daily"
    )
    mock_garmin_client.connectapi.return_value = [
        {
            "cycling": {
                "calendarDate": "2024-01-15",
                "vo2MaxValue": 55.1,
            }
        }
    ]

    result = await app_with_training.call_tool(
        "get_vo2max_trend", {"start_date": "2024-01-15", "end_date": "2024-01-15"}
    )

    data = json.loads(result[0][0].text)
    assert data["sport"] == "cycling"
    assert data["trend"] == [
        {
            "date": "2024-01-15",
            "vo2_max": 55.1,
            "source": "get_max_metrics",
        }
    ]
    mock_garmin_client.get_training_status.assert_not_called()
    mock_garmin_client.get_max_metrics.assert_not_called()


@pytest.mark.asyncio
async def test_get_vo2max_trend_range_failure_does_not_retry_max_metrics_daily(
    app_with_training, mock_garmin_client
):
    """Test a failed range request falls back without amplifying that failure"""
    mock_garmin_client.garmin_connect_metrics_url = (
        "/metrics-service/metrics/maxmet/daily"
    )
    mock_garmin_client.connectapi.side_effect = RuntimeError("API unavailable")
    mock_garmin_client.get_training_status.return_value = {
        "mostRecentVO2Max": {"generic": {"vo2MaxValue": 48.0}}
    }

    result = await app_with_training.call_tool(
        "get_vo2max_trend", {"start_date": "2024-01-14", "end_date": "2024-01-15"}
    )

    data = json.loads(result[0][0].text)
    assert data["sport"] == "running"
    assert data["source_failures"] == [
        {"source": "get_max_metrics", "kind": "connection"}
    ]
    assert mock_garmin_client.get_training_status.call_count == 2
    mock_garmin_client.get_max_metrics.assert_not_called()


@pytest.mark.asyncio
async def test_get_vo2max_trend_uses_daily_max_metrics_for_older_clients(
    app_with_training, mock_garmin_client
):
    """Test clients without the range API keep the compatible daily fallback"""
    mock_garmin_client.garmin_connect_metrics_url = None
    mock_garmin_client.get_training_status.return_value = {}
    mock_garmin_client.get_max_metrics.return_value = [
        {"generic": {"vo2MaxValue": 48.0}}
    ]

    result = await app_with_training.call_tool(
        "get_vo2max_trend", {"start_date": "2024-01-15", "end_date": "2024-01-15"}
    )

    data = json.loads(result[0][0].text)
    assert data["latest_vo2_max"] == 48.0
    mock_garmin_client.connectapi.assert_not_called()
    mock_garmin_client.get_max_metrics.assert_called_once_with("2024-01-15")


@pytest.mark.asyncio
async def test_get_vo2max_trend_prefers_running_when_sport_coverage_is_tied(
    app_with_training, mock_garmin_client
):
    """Test equal sport coverage has a stable running-first tie break"""
    mock_garmin_client.garmin_connect_metrics_url = (
        "/metrics-service/metrics/maxmet/daily"
    )
    mock_garmin_client.connectapi.return_value = []
    mock_garmin_client.get_training_status.return_value = {
        "mostRecentVO2Max": {
            "generic": {"vo2MaxValue": 48.0},
            "cycling": {"vo2MaxValue": 55.0},
        }
    }

    result = await app_with_training.call_tool(
        "get_vo2max_trend", {"start_date": "2024-01-15", "end_date": "2024-01-15"}
    )

    data = json.loads(result[0][0].text)
    assert data["sport"] == "running"
    assert data["latest_vo2_max"] == 48.0


@pytest.mark.asyncio
async def test_get_vo2max_trend_no_history_avoids_daily_max_metrics_requests(
    app_with_training, mock_garmin_client
):
    """Test the 90-day no-history case uses one range max-metrics request"""
    mock_garmin_client.garmin_connect_metrics_url = (
        "/metrics-service/metrics/maxmet/daily"
    )
    mock_garmin_client.connectapi.return_value = []
    mock_garmin_client.get_training_status.return_value = {}
    mock_garmin_client.get_user_profile.return_value = {
        "userData": {"vo2MaxRunning": 48.0}
    }

    result = await app_with_training.call_tool(
        "get_vo2max_trend", {"start_date": "2024-01-01", "end_date": "2024-03-30"}
    )

    data = json.loads(result[0][0].text)
    assert data["data_points"] == 0
    assert mock_garmin_client.get_training_status.call_count == 90
    assert mock_garmin_client.connectapi.call_count == 1
    mock_garmin_client.get_max_metrics.assert_not_called()
    mock_garmin_client.get_user_profile.assert_called_once_with()


@pytest.mark.asyncio
async def test_get_vo2max_trend_handles_endpoint_exception_and_missing_method(
    app_with_training, mock_garmin_client
):
    """Test daily errors and an unavailable max-metrics method reach the profile"""
    mock_garmin_client.garmin_connect_metrics_url = None
    mock_garmin_client.get_training_status.side_effect = RuntimeError("API unavailable")
    mock_garmin_client.get_max_metrics = None
    mock_garmin_client.get_user_profile.return_value = {
        "userData": {"vo2MaxCycling": 55.0}
    }

    result = await app_with_training.call_tool(
        "get_vo2max_trend", {"start_date": "2024-01-14", "end_date": "2024-01-15"}
    )

    data = json.loads(result[0][0].text)
    assert data["current_vo2_max_estimate"]["sport"] == "cycling"
    assert mock_garmin_client.get_training_status.call_count == 2


@pytest.mark.asyncio
async def test_get_lactate_threshold_tool_latest(app_with_training, mock_garmin_client):
    """Test get_lactate_threshold tool returns latest lactate threshold data"""
    # Setup mock with latest=True response format
    mock_garmin_client.get_lactate_threshold.return_value = MOCK_LACTATE_THRESHOLD

    # Call tool with no dates (gets latest)
    result = await app_with_training.call_tool(
        "get_lactate_threshold",
        {}
    )

    # Verify API call
    assert result is not None
    mock_garmin_client.get_lactate_threshold.assert_called_once_with(latest=True)

    # Verify output structure
    data = json.loads(result[0][0].text)
    assert data["lactate_threshold_speed_mps"] == 0.32222132
    assert data["lactate_threshold_heart_rate_bpm"] == 169
    assert data["functional_threshold_power_watts"] == 334
    assert data["sport"] == "RUNNING"
    assert data["power_to_weight"] == 4.575


@pytest.mark.asyncio
async def test_get_lactate_threshold_tool_range(app_with_training, mock_garmin_client):
    """Test get_lactate_threshold tool returns lactate threshold data for date range"""
    # Setup mock with date range response format
    mock_garmin_client.get_lactate_threshold.return_value = MOCK_LACTATE_THRESHOLD_RANGE

    # Call tool with date range
    result = await app_with_training.call_tool(
        "get_lactate_threshold",
        {"start_date": "2024-01-08", "end_date": "2024-01-15"}
    )

    # Verify API call
    assert result is not None
    mock_garmin_client.get_lactate_threshold.assert_called_once_with(
        latest=False,
        start_date="2024-01-08",
        end_date="2024-01-15",
    )

    # Verify output structure
    data = json.loads(result[0][0].text)
    assert data["start_date"] == "2024-01-08"
    assert data["end_date"] == "2024-01-15"
    assert "speed_history" in data
    assert len(data["speed_history"]) == 3
    assert data["speed_history"][0]["date"] == "2024-01-08"
    assert "heart_rate_history" in data
    assert len(data["heart_rate_history"]) == 3
    assert "power_history" in data


# Error handling tests
@pytest.mark.asyncio
async def test_get_hrv_data_no_data(app_with_training, mock_garmin_client):
    """Test get_hrv_data tool when no data available"""
    # Setup mock to return None
    mock_garmin_client.get_hrv_data.return_value = None

    # Call tool
    result = await app_with_training.call_tool(
        "get_hrv_data",
        {"date": "2024-01-15"}
    )

    # Verify error message is returned
    assert result is not None


@pytest.mark.asyncio
async def test_get_training_effect_exception(app_with_training, mock_garmin_client):
    """Test get_training_effect tool when API raises exception"""
    # Setup mock to raise exception - get_training_effect uses get_activity internally
    mock_garmin_client.get_activity.side_effect = Exception("API Error")

    # Call tool
    result = await app_with_training.call_tool(
        "get_training_effect",
        {"activity_id": 12345678901}
    )

    # Verify error is handled gracefully
    assert result is not None


@pytest.mark.asyncio
async def test_get_training_status_includes_cycling_vo2_max(app_with_training, mock_garmin_client):
    """Test that cycling VO2 max fields are surfaced when present in API response."""
    mock_garmin_client.get_training_status.return_value = MOCK_TRAINING_STATUS

    result = await app_with_training.call_tool(
        "get_training_status",
        {"date": "2024-01-15"},
    )

    assert result is not None
    import json
    text = result[0][0].text if result and result[0] else str(result)
    try:
        data = json.loads(text)
        assert data.get("cycling_vo2_max") == 55.0
        assert data.get("cycling_vo2_max_precise") == 55.12
    except (json.JSONDecodeError, AttributeError):
        # Tool may return raw text; just check the values appear in output
        assert "55.0" in text or "55.12" in text


@pytest.mark.asyncio
async def test_get_training_status_no_cycling_vo2_when_absent(app_with_training, mock_garmin_client):
    """Test that cycling VO2 fields are omitted when the cycling subkey is missing."""
    status_without_cycling = {
        "mostRecentVO2Max": {
            "generic": {"vo2MaxValue": 52.5, "vo2MaxPreciseValue": 52.47},
        },
    }
    mock_garmin_client.get_training_status.return_value = status_without_cycling

    result = await app_with_training.call_tool(
        "get_training_status",
        {"date": "2024-01-15"},
    )

    assert result is not None
    import json
    text = result[0][0].text if result and result[0] else str(result)
    try:
        data = json.loads(text)
        assert "cycling_vo2_max" not in data
        assert "cycling_vo2_max_precise" not in data
    except (json.JSONDecodeError, AttributeError):
        assert "cycling_vo2_max" not in text


def _training_status(
    calendar_date,
    *,
    devices=None,
    atl=250,
    ctl=220,
):
    """Synthetic training-status payload keyed by device id."""
    if devices is None:
        devices = [
            {
                "device_id": "111",
                "calendar_date": calendar_date,
                "primary": True,
                "atl": atl,
                "ctl": ctl,
            }
        ]
    latest = {}
    for device in devices:
        latest[device["device_id"]] = {
            "calendarDate": device.get("calendar_date", calendar_date),
            "primaryTrainingDevice": device.get("primary", False),
            "deviceId": device["device_id"],
            "trainingStatus": 1,
            "trainingStatusFeedbackPhrase": "MAINTAINING",
            "acuteTrainingLoadDTO": {
                "dailyTrainingLoadAcute": device.get("atl", atl),
                "dailyTrainingLoadChronic": device.get("ctl", ctl),
                "dailyAcuteChronicWorkloadRatio": 1.14,
            },
        }
    return {"mostRecentTrainingStatus": {"latestTrainingStatusData": latest}}


@pytest.mark.asyncio
async def test_training_load_trend_reports_missing_days_and_classified_failures(
    app_with_training, mock_garmin_client
):
    """T09: coverage and failures stay distinct; missing days are not filled with zero."""

    def by_date(date):
        if date == "2024-01-14":
            return _training_status("2024-01-14", atl=240, ctl=210)
        if date == "2024-01-15":
            return None
        if date == "2024-01-16":
            raise GarminConnectAuthenticationError("expired")
        if date == "2024-01-17":
            raise GarminConnectTooManyRequestsError("429")
        if date == "2024-01-18":
            raise GarminConnectConnectionError("timeout")
        raise AssertionError(date)

    mock_garmin_client.get_training_status.side_effect = by_date

    data = _tool_json(
        await app_with_training.call_tool(
            "get_training_load_trend",
            {"start_date": "2024-01-14", "end_date": "2024-01-18"},
        )
    )

    assert data["coverage"] == {
        "requested": 5,
        "available": 1,
        "missing": 1,
        "failed": 3,
        "stale": 0,
    }
    assert data["days_with_data"] == 1
    assert [row["date"] for row in data["trend"]] == ["2024-01-14"]
    assert data["trend"][0]["atl"] == 240.0
    kinds = {item["date"]: item["kind"] for item in data["failures"]}
    assert kinds == {
        "2024-01-16": "authentication",
        "2024-01-17": "rate_limit",
        "2024-01-18": "connection",
    }
    assert "2024-01-15" not in kinds


@pytest.mark.asyncio
async def test_training_load_trend_selects_primary_device_without_relabeling_stale_dates(
    app_with_training, mock_garmin_client
):
    """T10: keep the primary device series; do not stamp an older observation with the requested date."""

    def by_date(date):
        if date == "2024-01-14":
            return _training_status(
                date,
                devices=[
                    {
                        "device_id": "watch",
                        "calendar_date": "2024-01-14",
                        "primary": True,
                        "atl": 240,
                        "ctl": 210,
                    },
                    {
                        "device_id": "bike",
                        "calendar_date": "2024-01-14",
                        "primary": False,
                        "atl": 999,
                        "ctl": 888,
                    },
                ],
            )
        if date == "2024-01-15":
            return _training_status(
                date,
                devices=[
                    {
                        "device_id": "watch",
                        "calendar_date": "2024-01-10",
                        "primary": True,
                        "atl": 100,
                        "ctl": 200,
                    },
                    {
                        "device_id": "bike",
                        "calendar_date": "2024-01-15",
                        "primary": False,
                        "atl": 999,
                        "ctl": 888,
                    },
                ],
            )
        raise AssertionError(date)

    mock_garmin_client.get_training_status.side_effect = by_date

    data = _tool_json(
        await app_with_training.call_tool(
            "get_training_load_trend",
            {"start_date": "2024-01-14", "end_date": "2024-01-15"},
        )
    )

    assert data["coverage"] == {
        "requested": 2,
        "available": 1,
        "missing": 0,
        "failed": 0,
        "stale": 1,
    }
    assert data["stale"] == [
        {
            "requested_date": "2024-01-15",
            "observed_date": "2024-01-10",
            "device_id": "watch",
        }
    ]
    assert data["trend"] == [
        {
            "date": "2024-01-14",
            "device_id": "watch",
            "atl": 240.0,
            "ctl": 210.0,
            "tsb": -30.0,
            "acwr": 1.14,
            "training_status": "MAINTAINING",
            "training_status_code": 1,
        }
    ]


@pytest.mark.asyncio
async def test_training_load_uses_first_nonempty_device_as_fallback(
    app_with_training, mock_garmin_client
):
    payload = _training_status("2024-01-15")
    payload["mostRecentTrainingStatus"]["latestTrainingStatusData"] = {
        "empty": {},
        "watch": {
            "calendarDate": "2024-01-15",
            "deviceId": "watch",
            "acuteTrainingLoadDTO": {
                "dailyTrainingLoadAcute": 250,
                "dailyTrainingLoadChronic": 220,
            },
        },
    }
    mock_garmin_client.get_training_status.return_value = payload

    data = _tool_json(
        await app_with_training.call_tool(
            "get_training_load_trend",
            {"start_date": "2024-01-15", "end_date": "2024-01-15"},
        )
    )

    assert data["coverage"]["available"] == 1
    assert data["trend"][0]["device_id"] == "watch"
    assert data["trend"][0]["atl"] == 250.0


def _respiration(calendar_date, *, sleep=13.0, waking=14.0):
    return {
        "calendarDate": calendar_date,
        "avgWakingRespirationValue": waking,
        "avgSleepRespirationValue": sleep,
        "highestRespirationValue": 18.0,
        "lowestRespirationValue": 12.0,
    }


@pytest.mark.asyncio
async def test_respiration_trend_reports_missing_days_and_classified_failures(
    app_with_training, mock_garmin_client
):
    """T09: respiration coverage and failures stay distinct; missing days are not zeros."""

    def by_date(date):
        if date == "2024-01-14":
            return _respiration("2024-01-14", sleep=12.5)
        if date == "2024-01-15":
            return None
        if date == "2024-01-16":
            raise GarminConnectAuthenticationError("expired")
        if date == "2024-01-17":
            raise GarminConnectTooManyRequestsError("429")
        if date == "2024-01-18":
            raise GarminConnectConnectionError("timeout")
        raise AssertionError(date)

    mock_garmin_client.get_respiration_data.side_effect = by_date

    data = _tool_json(
        await app_with_training.call_tool(
            "get_respiration_trend",
            {"start_date": "2024-01-14", "end_date": "2024-01-18"},
        )
    )

    assert data["coverage"] == {
        "requested": 5,
        "available": 1,
        "missing": 1,
        "failed": 3,
        "stale": 0,
    }
    assert data["days_with_data"] == 1
    assert data["trend"][0]["date"] == "2024-01-14"
    assert data["trend"][0]["avg_sleep_breaths_per_min"] == 12.5
    kinds = {item["date"]: item["kind"] for item in data["failures"]}
    assert kinds == {
        "2024-01-16": "authentication",
        "2024-01-17": "rate_limit",
        "2024-01-18": "connection",
    }


@pytest.mark.asyncio
async def test_respiration_trend_skips_stale_calendar_date(
    app_with_training, mock_garmin_client
):
    """T10: an older respiration sample is not labeled as the requested day."""
    mock_garmin_client.get_respiration_data.return_value = _respiration(
        "2024-01-10", sleep=11.0
    )

    data = _tool_json(
        await app_with_training.call_tool(
            "get_respiration_trend",
            {"start_date": "2024-01-15", "end_date": "2024-01-15"},
        )
    )

    assert data["coverage"] == {
        "requested": 1,
        "available": 0,
        "missing": 0,
        "failed": 0,
        "stale": 1,
    }
    assert data["stale"] == [
        {
            "requested_date": "2024-01-15",
            "observed_date": "2024-01-10",
        }
    ]
    assert data["trend"] == []


@pytest.mark.asyncio
async def test_hrv_trend_reports_classified_failures_and_skips_stale_dates(
    app_with_training, mock_garmin_client
):
    """T09/T10: HRV failures are classified; a stale calendarDate is not relabeled."""

    def by_date(date):
        if date == "2024-01-14":
            return _hrv_payload(48, calendar_date="2024-01-14")
        if date == "2024-01-15":
            return _hrv_payload(40, calendar_date="2024-01-10")
        if date == "2024-01-16":
            raise GarminConnectConnectionError("timeout")
        raise AssertionError(date)

    mock_garmin_client.get_hrv_data.side_effect = by_date

    data = _tool_json(
        await app_with_training.call_tool(
            "get_hrv_trend",
            {"start_date": "2024-01-14", "end_date": "2024-01-16"},
        )
    )

    assert data["coverage"] == {
        "requested": 3,
        "available": 1,
        "missing": 0,
        "failed": 1,
        "stale": 1,
    }
    assert data["stale"] == [
        {
            "requested_date": "2024-01-15",
            "observed_date": "2024-01-10",
        }
    ]
    assert [row["date"] for row in data["trend"]] == ["2024-01-14"]
    assert data["trend"][0]["last_night_avg_hrv_ms"] == 48
    assert data["period_avg_hrv_ms"] == 48
    assert data["failures"] == [{"date": "2024-01-16", "kind": "connection"}]


@pytest.mark.asyncio
async def test_vo2max_trend_does_not_fabricate_history_from_stale_or_profile(
    app_with_training, mock_garmin_client
):
    """T10/T11: stale dated VO2 and the current profile stay out of the historical curve."""
    mock_garmin_client.garmin_connect_metrics_url = (
        "/metrics-service/metrics/maxmet/daily"
    )
    mock_garmin_client.connectapi.return_value = []
    mock_garmin_client.get_training_status.return_value = {
        "mostRecentVO2Max": {
            "generic": {"calendarDate": "2024-01-01", "vo2MaxValue": 48.0}
        }
    }
    mock_garmin_client.get_user_profile.return_value = {
        "userData": {"vo2MaxRunning": 28.0}
    }

    data = _tool_json(
        await app_with_training.call_tool(
            "get_vo2max_trend",
            {"start_date": "2024-01-14", "end_date": "2024-01-15"},
        )
    )

    assert data["coverage"] == {
        "requested": 2,
        "available": 0,
        "missing": 0,
        "failed": 0,
        "stale": 2,
    }
    assert data["stale"] == [
        {
            "requested_date": "2024-01-14",
            "observed_date": "2024-01-01",
        },
        {
            "requested_date": "2024-01-15",
            "observed_date": "2024-01-01",
        },
    ]
    assert data["data_points"] == 0
    assert data["trend"] == []
    assert data["current_vo2_max_estimate"] == {
        "vo2_max": 28.0,
        "sport": "running",
        "source": "get_user_profile",
    }


@pytest.mark.asyncio
async def test_vo2max_trend_classifies_daily_failures(
    app_with_training, mock_garmin_client
):
    """T09: a VO2 daily query failure is reported, not dropped."""
    mock_garmin_client.garmin_connect_metrics_url = None
    mock_garmin_client.get_max_metrics = None

    def by_date(date):
        if date == "2024-01-14":
            return {"mostRecentVO2Max": {"generic": {"vo2MaxValue": 48.0}}}
        if date == "2024-01-15":
            raise GarminConnectTooManyRequestsError("429")
        raise AssertionError(date)

    mock_garmin_client.get_training_status.side_effect = by_date

    data = _tool_json(
        await app_with_training.call_tool(
            "get_vo2max_trend",
            {"start_date": "2024-01-14", "end_date": "2024-01-15"},
        )
    )

    assert data["coverage"]["requested"] == 2
    assert data["coverage"]["available"] == 1
    assert data["coverage"]["failed"] == 1
    assert data["trend"][0]["vo2_max"] == 48.0
    assert data["failures"] == [{"date": "2024-01-15", "kind": "rate_limit", "source": "get_training_status"}]
    mock_garmin_client.get_user_profile.assert_not_called()


@pytest.mark.asyncio
async def test_vo2max_trend_keeps_stale_selected_sport_with_current_other_sport(
    app_with_training, mock_garmin_client
):
    mock_garmin_client.garmin_connect_metrics_url = None
    mock_garmin_client.get_max_metrics = None
    mock_garmin_client.get_training_status.side_effect = [
        {
            "mostRecentVO2Max": {
                "generic": {"calendarDate": "2024-01-14", "vo2MaxValue": 48.0}
            }
        },
        {
            "mostRecentVO2Max": {
                "generic": {"calendarDate": "2024-01-14", "vo2MaxValue": 48.0},
                "cycling": {"calendarDate": "2024-01-15", "vo2MaxValue": 55.0},
            }
        },
        {
            "mostRecentVO2Max": {
                "generic": {"calendarDate": "2024-01-16", "vo2MaxValue": 49.0}
            }
        },
    ]

    data = _tool_json(
        await app_with_training.call_tool(
            "get_vo2max_trend",
            {"start_date": "2024-01-14", "end_date": "2024-01-16"},
        )
    )

    assert data["sport"] == "running"
    assert data["coverage"] == {
        "requested": 3,
        "available": 2,
        "missing": 0,
        "failed": 0,
        "stale": 1,
    }
    assert data["stale"] == [
        {"requested_date": "2024-01-15", "observed_date": "2024-01-14"}
    ]


@pytest.mark.asyncio
async def test_vo2max_trend_filters_dates_independently_by_sport(
    app_with_training, mock_garmin_client
):
    mock_garmin_client.garmin_connect_metrics_url = None
    mock_garmin_client.get_training_status.return_value = {}
    mock_garmin_client.get_max_metrics.return_value = {
        "generic": {"vo2MaxValue": 48.0},
        "cycling": {"calendarDate": "2024-01-14", "vo2MaxValue": 55.0},
    }

    data = _tool_json(
        await app_with_training.call_tool(
            "get_vo2max_trend",
            {"start_date": "2024-01-15", "end_date": "2024-01-15"},
        )
    )

    assert data["sport"] == "running"
    assert data["trend"][0]["vo2_max"] == 48.0
    assert data["coverage"]["available"] == 1


@pytest.mark.asyncio
async def test_vo2max_trend_reports_stale_list_response(
    app_with_training, mock_garmin_client
):
    mock_garmin_client.garmin_connect_metrics_url = None
    mock_garmin_client.get_training_status.return_value = {}
    mock_garmin_client.get_max_metrics.return_value = [
        {
            "generic": {
                "calendarDate": "2024-01-14",
                "vo2MaxValue": 48.0,
            }
        }
    ]
    mock_garmin_client.get_user_profile.return_value = {}

    data = _tool_json(
        await app_with_training.call_tool(
            "get_vo2max_trend",
            {"start_date": "2024-01-15", "end_date": "2024-01-15"},
        )
    )

    assert data["coverage"]["missing"] == 0
    assert data["coverage"]["stale"] == 1
    assert data["stale"] == [
        {"requested_date": "2024-01-15", "observed_date": "2024-01-14"}
    ]


@pytest.mark.asyncio
async def test_trends_isolate_malformed_days_during_payload_processing(
    app_with_training, mock_garmin_client
):
    mock_garmin_client.get_hrv_data.side_effect = [
        _hrv_payload(48, calendar_date="2024-01-14", weekly_avg="N/A"),
        _hrv_payload(49, calendar_date="2024-01-15", weekly_avg=46),
    ]
    hrv = _tool_json(
        await app_with_training.call_tool(
            "get_hrv_trend",
            {"start_date": "2024-01-14", "end_date": "2024-01-15"},
        )
    )
    assert hrv["coverage"]["available"] == 1
    assert hrv["failures"] == [{"date": "2024-01-14", "kind": "connection"}]

    malformed_load = _training_status("2024-01-14", atl="bad", ctl=220)
    mock_garmin_client.get_training_status.side_effect = [
        malformed_load,
        _training_status("2024-01-15", atl=250, ctl=220),
    ]
    load = _tool_json(
        await app_with_training.call_tool(
            "get_training_load_trend",
            {"start_date": "2024-01-14", "end_date": "2024-01-15"},
        )
    )
    assert load["coverage"]["available"] == 1
    assert load["failures"] == [{"date": "2024-01-14", "kind": "connection"}]

    mock_garmin_client.get_respiration_data.side_effect = [
        _respiration("2024-01-14", sleep="bad"),
        _respiration("2024-01-15", sleep=13.0),
    ]
    respiration = _tool_json(
        await app_with_training.call_tool(
            "get_respiration_trend",
            {"start_date": "2024-01-14", "end_date": "2024-01-15"},
        )
    )
    assert respiration["coverage"]["available"] == 1
    assert respiration["failures"] == [
        {"date": "2024-01-14", "kind": "connection"}
    ]


@pytest.mark.asyncio
async def test_load_trend_keeps_current_load_but_reports_stale_vo2(app_with_training, mock_garmin_client):
    def by_date(date):
        payload = _training_status(date)
        payload['mostRecentVO2Max'] = {
            'generic': {'calendarDate': '2024-01-01', 'vo2MaxValue': 48.0}
        }
        return payload

    mock_garmin_client.get_training_status.side_effect = by_date
    data = _tool_json(await app_with_training.call_tool('get_training_load_trend', {
        'start_date': '2024-01-14', 'end_date': '2024-01-15',
    }))
    assert len(data['trend']) == 2
    assert all('atl' in point and 'vo2_max' not in point for point in data['trend'])
    assert data['coverage'] == {'requested': 2, 'available': 2, 'missing': 0, 'failed': 0, 'stale': 2}
    assert all(item['observed_date'] == '2024-01-01' and item['metric'] == 'vo2_max' for item in data['stale'])


@pytest.mark.asyncio
async def test_load_trend_keeps_vo2_observed_on_requested_day(app_with_training, mock_garmin_client):
    payload = _training_status('2024-01-14')
    payload['mostRecentVO2Max'] = {'generic': {'calendarDate': '2024-01-14', 'vo2MaxValue': 48.0}}
    mock_garmin_client.get_training_status.return_value = payload
    data = _tool_json(await app_with_training.call_tool('get_training_load_trend', {
        'start_date': '2024-01-14', 'end_date': '2024-01-14',
    }))
    assert data['trend'][0]['vo2_max'] == 48.0
    assert data['stale'] == []


@pytest.mark.asyncio
@pytest.mark.parametrize('fallback', [None, {}, {'generic': {'vo2MaxValue': 48.0, 'calendarDate': '2024-01-01'}}, {'generic': {'vo2MaxValue': 48.0, 'calendarDate': '2024-01-14'}}])
async def test_vo2_trend_preserves_failure_when_fallback_responds(app_with_training, mock_garmin_client, fallback):
    mock_garmin_client.garmin_connect_metrics_url = None
    mock_garmin_client.get_training_status.side_effect = GarminConnectTooManyRequestsError('429')
    mock_garmin_client.get_max_metrics.return_value = fallback
    mock_garmin_client.get_user_profile.return_value = {}
    data = _tool_json(await app_with_training.call_tool('get_vo2max_trend', {
        'start_date': '2024-01-14', 'end_date': '2024-01-14',
    }))
    assert data['failures'] == [{'date': '2024-01-14', 'kind': 'rate_limit', 'source': 'get_training_status'}]
    assert data['coverage']['failed'] == 1
    assert data['coverage']['missing'] == 0
    assert data['coverage']['available'] == int(bool(data['trend']))
