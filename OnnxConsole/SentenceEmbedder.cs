using BERTTokenizers;
using FaissSharp;
using Microsoft.Extensions.Primitives;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static System.Collections.Specialized.BitVector32;

namespace OnnxConsole
{
    internal class SentenceEmbedder
    {
        public string[] texts = { "Antgua Racer is a very fast reptile and highly agile" };//, "Sloths are slow", "Anaconda is a large snake and moves slowly" };
        BertBaseTokenizer tokenizer;
        String modelPath;
        InferenceSession session;
        public SentenceEmbedder() {
             tokenizer = new BertBaseTokenizer();
             modelPath = "C:\\Users\\marjankhan945\\Downloads\\all-mpnet-base-v2.onnx";
             session = new InferenceSession(modelPath);


           



             //faissIndex = new IndexFlat(d); //TODO: FAISS for windows needs to be installed and currently need conda
             // Try with cmake if custom build can be done to provide with package

        }

        List<InternalTexData> textss = new List<InternalTexData>();

        public  IReadOnlyList<TextData> Query(string text)
        {
            var textData = new TextData();
            textData.text = text;
            var intTextData = new InternalTexData(textData);
            GetEmbeddingForSentence(intTextData);

            List<TextData> res = new List<TextData>();
            foreach(var data in textss)
            {
                data.similarity = CosineSimilarity(data.embedding, intTextData.embedding);
                res.Add(data);
            }

            return res;
        }

        public void SentencesToIndex(List<TextData> textDatas)
        {
            foreach (var textData in textDatas)
            {
                var x = new InternalTexData(textData);
                textss.Add(x);
                GetEmbeddingForSentence(x);
            }
        }

        private void GetEmbeddingForSentence(in InternalTexData internalTex)
        {
            var tokesn = tokenizer.Tokenize(internalTex.text);
            var ecnoded = tokenizer.Encode(512, internalTex.text);


            var modelInp = new ModelInput()
            {
                AttentionMask = ecnoded.Select(t => t.AttentionMask).ToArray(),
                InputIds = ecnoded.Select(t => t.InputIds).ToArray()
            };

            Tensor<long> inputIds = ToTensorTYpe(modelInp.InputIds, modelInp.InputIds.Length);
            Tensor<long> attentionMask = ToTensorTYpe(modelInp.AttentionMask, modelInp.AttentionMask.Length);



            var input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<long>("input_ids", inputIds),
                                                   NamedOnnxValue.CreateFromTensor<long>("attention_mask", attentionMask)};


            var outputs = session.Run(input);

            var str1384 = (outputs.ToList().First().Value as IEnumerable<float>).ToList();


            internalTex.embedding = str1384;// MeanPooling(str1384, tokesn.Count());
        }

        private static List<float> MeanPooling(List<float> embedding, int tokenSize)
        {
            List<float> pooled = new List<float>();
            int n = embedding.Count / tokenSize; 

            for (int i = 0; i < n; i++)
            {
                float sum = 0;
                for (int j = 0; j < 4; j++)
                {
                    sum += embedding[i * 4 + j];
                }
                float mean = sum / 4.0f;
                pooled.Add(mean);
            }

            return pooled;
        }

        public static Tensor<long> ToTensorTYpe(long[] inputArray, int inputDim)
        {
            Tensor<long> input = new DenseTensor<long>(new[] { 1, inputDim });
            for (var i = 0; i < inputArray.Length; i++)
            {
                input[0, i] = inputArray[i];
            }

            return input;
        }

        private static double CosineSimilarity(List<float> A, List<float> B)
        {
            double dotProduct = A.Zip(B, (x, y) => (double)x * y).Sum();

            double mag1 = Math.Sqrt(A.Select(x => (double)x * x).Sum());
            double mag2 = Math.Sqrt(B.Select(x => (double)x * x).Sum());

            double sim = dotProduct / (mag1 * mag2);    
            return sim;
        }
    }

    internal class InternalTexData : TextData
    {
       public  List<float> embedding;
        public InternalTexData(TextData data)
        {
            this.text = data.text;
            this.datas = data.datas;
            this.similarity = 0.0f;
            this.embedding = new List<float>();
        }
    }

    public class DataBaseFormat
    {
        public Dictionary<string, object> datas; // keep relevant data of sentence to use afer result is received in return
        public double similarity;
    }

    public class TextData : DataBaseFormat
    {
        public string text;
    }

    public class ModelInput
    {
        [VectorType(1, 512)]
        [ColumnName("input_ids")]
        public long[] InputIds { get; set; }

        [VectorType(1, 512)]
        [ColumnName("attention_mask")]
        public long[] AttentionMask { get; set; }
    }

    public class ModelOutput
    {
        [VectorType(1, 512, 768)]
        [ColumnName("onnx::Gather_1381")]
        public long[] LastHiddenState { get; set; }

        [VectorType(1, 768)]
        [ColumnName("1384")]
        public long[] PollerOutput { get; set; }
    }
}
