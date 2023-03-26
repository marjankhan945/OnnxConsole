// See https://aka.ms/new-console-template for more information
using OnnxConsole;

Console.WriteLine("Hello, World!");

var embedder = new SentenceEmbedder();


 List<TextData> sentToIndex = new List<TextData>
{
    new TextData()
    {
        text = "There was a man in the jungle who lived alone"
    },
    new TextData()
    {
        text = "Black mamba is a widely known snake for being venomous"
    },
    new TextData()
    {
        text = "Cobra is a very fast snake and is highly venomous"
    },
    new TextData()
    {
        text = "Anaconda is a very slow snake and they are very large in size"
    },
     new TextData()
    {
        text = "Turn on ambient sound"
    },
     new TextData()
    {
        text = "Turn on noise cancellation feature"
    },
     new TextData()
    {
        text = "Turn on voice detection feature"
    },
};

embedder.SentencesToIndex(sentToIndex);

var res = embedder.Query("snake that is fast");

string nothing = res.MaxBy(x => x.similarity).text;
Console.WriteLine(nothing);

res = embedder.Query("reptlie that moves sluggishly");

 nothing = res.MaxBy(x => x.similarity).text;
Console.WriteLine(nothing);