# Amazon Reviews using OpenAI embeddings and ML.NET AutoML

.NET Console app that shows how to train a regression model to predict the score using ML.NET AutoML and Azure OpenAI service embeddings for fine food reviews.

## Dataset

[Amazon Fine Foods Review Dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews?resource=download) - ~500,000 food reviews from Amazon. 

Id | ProductId | UserId | ProfileName | HelpfulnessNumerator | HelpfulnessDenominator | Score | Time | Summary | Text
| --- |--- |--- |--- |--- |--- |--- |--- |--- |--- |
|1|B001E4KFG0|A3SGXH7AUHU8GW|delmartian|1|1|5|1303862400|Good Quality Dog Food|I have bought several of the Vitality canned dog food products and have found them all to be of good quality. The product looks more like a stew than a processed meat and it smells better. My Labrador is finicky and she appreciates this product better than  most.
|2|B00813GRG4|A1D87F6ZCVE5NK|dll pa|0|0|1|1346976000|Not as Advertised|"Product arrived labeled as Jumbo Salted Peanuts...the peanuts were actually small sized unsalted. Not sure if this was an error or if the vendor intended to represent the product as ""Jumbo""."
|3|B000LQOCH0|ABXLMWJIXXAIN|"Natalia Corres ""Natalia Corres"""|1|1|4|1219017600|"""Delight"" says it all"|"This is a confection that has been around a few centuries.  It is a light, pillowy citrus gelatin with nuts - in this case Filberts. And it is cut into tiny squares and then liberally coated with powdered sugar.  And it is a tiny mouthful of heaven.  Not too chewy, and very flavorful.  I highly recommend this yummy treat.  If you are familiar with the story of C.S. Lewis' ""The Lion, The Witch, and The Wardrobe"" - this is the treat that seduces Edmund into selling out his Brother and Sisters to the Witch."


## Prerequisites

- [.NET 7 SDK](https://dotnet.microsoft.com/en-us/download/dotnet/7.0)
- [Azure Subscription](aka.ms/free)
- [Azure OpenAI Service](https://learn.microsoft.com/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal)
    - Open AI Model Deployed (**text-similarity-curie-001**).

## Instructions

1. Download the [dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews?resource=download)
1. Unzip the dataset and save the *Reviews.csv* file to to the *data* directory.
1. Set the following environment variables using information from your Azure Open AI service resource:
    - **OPENAI_ENDPOINT** - Azure OpenAI service endpoint.
    - **OPENAI_KEY** - Azure OpenAI service access key.
    - **OPENAI_DEPLOYMENT** - The name of the model deployment.
1. Build the project

    ```dotnetcli
    dotnet build
    ```

1. Run the project

    ```dotnetcli
    dotnet run
    ```

If successful, you should see output similar to the following:

```text
Trial 0 finished training in 331016ms with pipeline FastForestRegression
Test Set MAE: 0.40577530734082484
```