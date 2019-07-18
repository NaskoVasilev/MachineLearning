using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Threading.Tasks;

namespace WordsClassification
{
    public class DataGenerator
    {
        public async Task<IEnumerable<WordModel>> GenerateData()
        {
            string baseApiUrlAddress = "https://api.datamuse.com/";
            string apiQueryString = "words?topics={0}&max=80";
            var words = new List<WordModel>();
            IEnumerable<string> categories = GetCategories();

            HttpClient httpClient = new HttpClient { BaseAddress = new Uri(baseApiUrlAddress) };

            foreach (var category in categories)
            {
                string path = string.Format(apiQueryString, category);
                string json = await httpClient.GetStringAsync(path);
                var currentWords = JsonConvert.DeserializeObject<IEnumerable<WordModel>>(json);

                foreach (var currentWord in currentWords.Where(w => !w.Word.Contains(' ')))
                {
                    currentWord.Category = category;
                    words.Add(currentWord);
                }
            }

            return words;
        }

        private IEnumerable<string> GetCategories()
        {
            string filePath = "../../../Resources/WordCategories.txt";
            string text = File.ReadAllText(filePath).ToLower();
            return text.Split(", ");
        }
    }
}
