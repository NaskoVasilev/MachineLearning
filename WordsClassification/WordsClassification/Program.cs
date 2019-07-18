using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace WordsClassification
{
    class Program
    {
        static void Main(string[] args)
        {
            DataGenerator generator = new DataGenerator();
            var data = generator.GenerateData().GetAwaiter().GetResult();

            string dataFilePath = "../../../words-train-data.csv";
            CreateDataFile(dataFilePath, data);

            var modelFile = "../../../WordsCategoryModel.zip";
            if (!File.Exists(modelFile))
            {
                TrainModel(dataFilePath, modelFile);
            }

            var testData = new List<string>(){ "cat", "tennis", "building", "painting","home",
                "shoes", "automobile", "brother", "waterfall", "occupation" };
            TestModel(modelFile, testData);
        }

        private static void CreateDataFile(string filePath, IEnumerable<WordModel> data)
        { 
            if(File.Exists(filePath))
            {
                return;
            }
            var lines = new List<string>();
            lines.Add("Category,Word");
            foreach (var word in data)
            {
                lines.Add($"{word.Category},{word.Word}");
            }

            File.WriteAllLines(filePath, lines);
        }

        private static void TrainModel(string dataFile, string modelFile)
        {
            // Create MLContext to be shared across the model creation workflow objects
            var context = new MLContext(seed: 0);

            // Loading the data
            Console.WriteLine($"Loading the data ({dataFile})");
            var trainingDataView = context.Data.LoadFromTextFile<WordModel>(dataFile, ',', true, true, true);

            // Common data process configuration with pipeline data transformations
            Console.WriteLine("Map raw input data columns to ML.NET data");
            var dataProcessPipeline = context.Transforms.Conversion.MapValueToKey("Label", nameof(WordModel.Category))
                .Append(context.Transforms.Text.FeaturizeText("Features", nameof(WordModel.Word)));

            // Create the selected training algorithm/trainer
            Console.WriteLine("Create and configure the selected training algorithm (trainer)");
            var trainer = context.MulticlassClassification.Trainers.SdcaMaximumEntropy(); // SDCA = Stochastic Dual Coordinate Ascent
            //// Alternative: LightGbm (GBM = Gradient Boosting Machine)

            // Set the trainer/algorithm and map label to value (original readable state)
            var trainingPipeline = dataProcessPipeline.Append(trainer).Append(
                context.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // Train the model fitting to the DataSet
            Console.WriteLine("Train the model fitting to the DataSet");
            var trainedModel = trainingPipeline.Fit(trainingDataView);

            // Save/persist the trained model to a .ZIP file
            Console.WriteLine($"Save the model to a file ({modelFile})");
            context.Model.Save(trainedModel, trainingDataView.Schema, modelFile);
        }

        private static void TestModel(string modelFile, IEnumerable<string> testModelData)
        {
            var context = new MLContext();
            var model = context.Model.Load(modelFile, out _);
            var predictionEngine = context.Model.CreatePredictionEngine<WordModel, WordModelPrediction>(model);
            foreach (var testData in testModelData)
            {
                var prediction = predictionEngine.Predict(new WordModel { Word = testData });
                Console.WriteLine(new string('-', 60));
                Console.WriteLine($"Content: {testData}");
                Console.WriteLine($"Prediction: {prediction.Category}");
            }
        }
    }
}
