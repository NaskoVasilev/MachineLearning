using Microsoft.ML;
using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;

namespace JudgeSystemLessonsRecomendation
{
    public class Program
    {
        private const string ModelFile = "JudgeSystemLessonsModel.zip";
        private const string DataFile = "judge-system-user-lessons.csv";

        static void Main(string[] args)
        {
            TrainModel(DataFile, ModelFile);

            var testModelData = new List<UserLesson>
            {
                // UserId = 245fb250-9d69-4e5c-8026-b1253eb1c728 => Multidimensional Arrays, Trees, Binary Search Trees, Polymorphism, Reflection, Recursion
                //Sorting and Searching, Combinatorial Algoritms, Graphs and Graph Algorithms,  Advanced Graph Algorithms
                new UserLesson { UserId = "ee14a184-7282-49b8-bd9a-ade05693f6e4", LessonId = 26 }, // Simple Operations and Calculations
                new UserLesson { UserId = "ee14a184-7282-49b8-bd9a-ade05693f6e4", LessonId = 38 }, // Data Types and Variables

                new UserLesson { UserId = "ee14a184-7282-49b8-bd9a-ade05693f6e4", LessonId = 51 }, // Sets and Dictionaries
                new UserLesson { UserId = "ee14a184-7282-49b8-bd9a-ade05693f6e4", LessonId = 81 }, // Attributes
                new UserLesson { UserId = "ee14a184-7282-49b8-bd9a-ade05693f6e4", LessonId = 75 }, // Interfaces
                new UserLesson { UserId = "ee14a184-7282-49b8-bd9a-ade05693f6e4", LessonId = 63 }, // Projections
            };

            TestModel(ModelFile, testModelData);
        }

        private static void TrainModel(string inputFile, string modelFile)
        {
            // Create MLContext to be shared across the model creation workflow objects
            var context = new MLContext();

            // Load data
            IDataView trainingDataView = context.Data.LoadFromTextFile<UserLesson>(
                inputFile,
                hasHeader: true,
                separatorChar: ',');

            // Build & train model
            IEstimator<ITransformer> estimator = context.Transforms.Conversion
                .MapValueToKey(outputColumnName: "userIdEncoded", inputColumnName: nameof(UserLesson.UserId)).Append(
                    context.Transforms.Conversion.MapValueToKey(outputColumnName: "lessonIdEncoded", inputColumnName: nameof(UserLesson.LessonId)));
            var options = new MatrixFactorizationTrainer.Options
            {
                LossFunction = MatrixFactorizationTrainer.LossFunctionType.SquareLossOneClass,
                MatrixColumnIndexColumnName = "userIdEncoded",
                MatrixRowIndexColumnName = "lessonIdEncoded",
                LabelColumnName = nameof(UserLesson.Label),
                Alpha = 0.1,
                Lambda = 0.5,
                NumberOfIterations = 50,
            };

            var trainerEstimator = estimator.Append(context.Recommendation().Trainers.MatrixFactorization(options));
            ITransformer model = trainerEstimator.Fit(trainingDataView);

            // Save model
            context.Model.Save(model, trainingDataView.Schema, modelFile);
        }

        private static void TestModel(string modelFile, IEnumerable<UserLesson> testModelData)
        {
            var context = new MLContext();
            var model = context.Model.Load(modelFile, out _);
            var predictionEngine = context.Model.CreatePredictionEngine<UserLesson, UserLessonScore>(model);
            foreach (var testInput in testModelData)
            {
                var prediction = predictionEngine.Predict(testInput);
                Console.WriteLine($"User: {testInput.UserId}, Lesson: {testInput.LessonId}, Score: {prediction.Score}");
            }
        }
    }
}
