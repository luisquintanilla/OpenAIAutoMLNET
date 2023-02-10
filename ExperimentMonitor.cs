using Microsoft.ML.AutoML;

public class ExperimentMonitor : IMonitor
{
    private readonly SweepablePipeline _pipeline;
    private readonly List<TrialResult> _completedTrials;
    public ExperimentMonitor(SweepablePipeline pipeline)
    {
        _pipeline = pipeline;
        _completedTrials = new List<TrialResult>();
    }

    public IEnumerable<TrialResult> GetCompletedTrials() => _completedTrials;

    public void ReportBestTrial(TrialResult result)
    {
        return;
    }

    public void ReportCompletedTrial(TrialResult result)
    {
        var trialId = result.TrialSettings.TrialId;
        var timeToTrain = result.DurationInMilliseconds;
        var pipeline = _pipeline.ToString(result.TrialSettings.Parameter);
        _completedTrials.Add(result);
        Console.WriteLine($"Trial {trialId} finished training in {timeToTrain}ms with pipeline {pipeline}");
    }

    public void ReportFailTrial(TrialSettings settings, Exception exception = null)
    {
        if (exception.Message.Contains("Operation was canceled."))
        {
            Console.WriteLine($"{settings.TrialId} cancelled. Time budget exceeded.");
        }
        Console.WriteLine($"{settings.TrialId} failed with exception {exception.Message}");
    }

    public void ReportRunningTrial(TrialSettings setting)
    {
        return;
    }
}