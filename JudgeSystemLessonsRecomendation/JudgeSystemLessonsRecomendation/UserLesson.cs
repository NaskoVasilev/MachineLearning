using Microsoft.ML.Data;

namespace JudgeSystemLessonsRecomendation
{
    public class UserLesson
    {
        [LoadColumn(0)]
        public string UserId { get; set; }

        [LoadColumn(1)]
        public int LessonId { get; set; }

        //[LoadColumn(2)]
        //public float Lebel { get; set; }
    }
}
