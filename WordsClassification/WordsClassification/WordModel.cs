using Microsoft.ML.Data;

namespace WordsClassification
{
    public class WordModel
    {
        [LoadColumn(0)]
        public string Category { get; set; }

        [LoadColumn(1)]
        public string Word { get; set; }
    }
}
