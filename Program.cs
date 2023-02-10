// Add using statements
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.AutoML;
using Azure.AI.OpenAI;

// Initialize Azure OpenAI client
var endpoint = new Uri(Environment.GetEnvironmentVariable("OPENAI_ENDPOINT"));
var key = new Azure.AzureKeyCredential(Environment.GetEnvironmentVariable("OPENAI_KEY")); 
var deploymentId = Environment.GetEnvironmentVariable("OPENAI_DEPLOYMENT");
var client = new OpenAIClient(endpoint, key);

// Initialize MLContext
var ctx = new MLContext();

// Get column information
// Dataset https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews?resource=download
var inferResults = ctx.Auto().InferColumns("data/Reviews.csv", hasHeader: true, labelColumnIndex:6, separatorChar:',', groupColumns:false);

// Load data
var textLoader = ctx.Data.CreateTextLoader(inferResults.TextLoaderOptions);
var data = textLoader.Load("data/Reviews.csv");

// Drop columns
var dropOperation = ctx.Transforms.DropColumns(new [] {"Id", "ProductId", "UserId","ProfileName", "HelpfulnessNumerator", "HelpfulnessDenominator"});
var droppedColsDv = dropOperation.Fit(data).Transform(data);

// Get most recent reviews (lower bound is unix timestamp for Jan 1, 2012)
var mostRecentDv = ctx.Data.FilterRowsByColumn(droppedColsDv, "Time", lowerBound: 1325394000.0);

// Shuffle data
var shuffledData = ctx.Data.ShuffleRows(mostRecentDv);

// Create a subset of 500 rows
var dataSubset = ctx.Data.TakeRows(shuffledData, 500);

// Custom transform to get OpenAI embeddings
var getEmbedding = (EmbeddingInput input, EmbeddingOutput output) => 
{
    // Initialize embedding options using text from PreembedText column
    var embeddingOptions = new EmbeddingsOptions(input.PreembedText);

    // Get embeddings of PreembedText column
    Embeddings embeddingResult = client.GetEmbeddings(deploymentId, embeddingOptions);

    // Store embeddings in Embeddings column
    output.Embeddings = embeddingResult.Data[0].Embedding.ToArray();
};

// Prepare data
var dataPrepPipeline = 
    ctx.Transforms.CopyColumns("Label","Score") // Create column called label
    .Append(ctx.Transforms.DropColumns("Score")) // Drop original Score column
    .Append(ctx.Transforms.Expression(
        outputColumnName: "PreembedText",
        expression: "(summary,text) => left(concat(summary,text), 4096)",
        inputColumnNames: new []{"Summary", "Text"})) // Concatenate summary and text columns and take first 4096 elements
    .Append(ctx.Transforms.CustomMapping(getEmbedding, "GetEmbedding")); // Get OpenAI embeddings

// Apply preprocessing transforms
var preprocessedData = 
    dataPrepPipeline.Fit(dataSubset).Transform(dataSubset);

// Split data into train (400 reviews) / test (100 reviews) sets
var trainTestData = ctx.Data.TrainTestSplit(preprocessedData, testFraction: 0.2);

// Define AutoML Task
var regressionPipeline = ctx.Auto().Regression(labelColumnName: "Label", featureColumnName: "Embeddings", useLgbm: false);

// Initialize Experiment
var experiment = ctx.Auto().CreateExperiment();

// Configure experiment
// This may take longer than 180 seconds
experiment
    .SetDataset(trainTestData.TrainSet)
    .SetPipeline(regressionPipeline)
    .SetRegressionMetric(RegressionMetric.MeanAbsoluteError)
    .SetTrainingTimeInSeconds(180);

// Configure monitor
var monitor = new ExperimentMonitor(regressionPipeline);
experiment.SetMonitor<ExperimentMonitor>(monitor);

// Run experiment
var experimentResult = await experiment.RunAsync();

// Get model
var model = experimentResult.Model;

// Use model to make predictions on test dataset
var predictions = model.Transform(trainTestData.TestSet);

// Calculate evaluation metrics on test dataset
var testEvaluationMetrics = ctx.Regression.Evaluate(predictions); 

// Output Mean Absoute Error
Console.WriteLine($"Test Set MAE: {testEvaluationMetrics.MeanAbsoluteError}");

public class EmbeddingInput
{
    public string PreembedText { get; set; }

}

public class EmbeddingOutput
{
    public string PreembedText { get; set; }
    
    [VectorType(4096)]
    public float[] Embeddings { get; set; }
}