using Microsoft.ML.Data;

namespace WordsClassification
{
    public class WordModelPrediction
    {
        [ColumnName("PredictedLabel")]
        public string Category { get; set; }
    }
}
