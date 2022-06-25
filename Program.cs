
public interface IActivation
{
    Func<double, double> Activation { get; set; }
    Func<double, double> Derivative { get; set; }

}

public class Sigmoid : IActivation
{
    public Func<double, double> Activation
    {
        get
        {
            return new Func<double, double>((x) => { return 1.0 / (1.0 + Math.Exp(-x)); });
        }
        set { }
    }
    public Func<double, double> Derivative
    {
        get
        {
            return new Func<double, double>((x) =>
            {
                return Activation(x) * (1.0 - Activation(x));
            });
        }
        set { }
    }
}


public class Linear : IActivation
{
    public Func<double, double> Activation
    {
        get
        {
            return new Func<double, double>((x) => { return x; });
        }
        set { }
    }
    public Func<double, double> Derivative
    {
        get
        {
            return new Func<double, double>((x) =>
            {
                return 1.0d;
            });
        }
        set { }
    }
}

public class Neurone
{

    public IActivation Activation { get; set; }
    public List<Connection> Connections { get; set; }
    public double WeightedInput { get; set; }
    public double Bias { get; set; }
    public double Delta { get; private set; }

    public Neurone(IActivation activation)
    {

        Activation = activation;
        Connections = new List<Connection>();
    }
    public void Adjust(double delta)
    {
        Bias += delta;
    }

    public void Fire() { Connections.ForEach(p => p.Fire()); }
    public void Clear()
    {
        WeightedInput = 0;
    }

    public void Set(double value) { WeightedInput += value; }
    public double Get() { return Activation.Activation(WeightedInput + Bias); }

    public void CalculateDelta(double requestedValue)
    {
        Delta = Activation.Derivative(WeightedInput + Bias) * (requestedValue - Get());
    }
}

public class Connection
{
    public Neurone Dendrite { get; set; }
    public Neurone Axon { get; set; }

    public double Weight { get; set; }

    public Connection(Neurone a, Neurone b, double weight)
    {
        Dendrite = a;
        Axon = b;
        Weight = weight;
    }

    double lastWeight = 1;
    public void Adjust(double delta, double momentum)
    {
        Weight += delta + (lastWeight * momentum);
        lastWeight = delta;
    }
    public void Fire()
    {
        Axon.Set(Dendrite.Get() * Weight);
    }
}

public class SimpleNet
{
    public Neurone[] Inputs; public Neurone Output;
    public Dictionary<double[], double> Trainingset = new Dictionary<double[], double>();
    public SimpleNet()
    {
        Inputs = new Neurone[2];
        for (int i = 0; i < Inputs.Length; i++)
            Inputs[i] = new Neurone(new Linear());

        Output = new Neurone(new Sigmoid()) { Bias = Program.R.NextDouble() };

        for (int i = 0; i < Inputs.Length; i++)
            Inputs[i].Connections.Add(new Connection(Inputs[i], Output, Program.R.NextDouble() * 0.1));
    }

    public double Forward(double[] inputs)
    {
        for (int i = 0; i < Inputs.Length; i++)
            Inputs[i].Set(inputs[i]);

        for (int j = 0; j < Inputs.Length; j++)
            Inputs[j].Fire();

        return Output.Get();
    }


    public void Backpropagate(double epsilon, double accuraccy, int iterations = -1)
    {

        int epoch = 0;

        while (true)
        {
            double error = 0;

            foreach (var data in Trainingset)
            {
                Forward(data.Key); Output.CalculateDelta(data.Value);
                for (int j = 0; j < Inputs.Length; j++)
                {
                    Inputs[j].Connections[0].Adjust(Output.Delta * epsilon * Inputs[j].Get(), 0.8);
                    Output.Adjust(Output.Delta * epsilon);
                    error += Math.Pow(Math.Abs(data.Value - Output.Get()), 2);
                    Clear();
                }

                error *= 0.5; error = 1.0 - error; Console.WriteLine("Accuracy:" + error);
                if (iterations == -1)
                    if (error >= accuraccy)
                        return;
                    else if (epoch++ >= iterations)
                        return;
            }
        }
    }
    public void Clear()
    {
        Output.Clear(); for (int i = 0; i < Inputs.Length; i++)
            Inputs[i].Clear();
    }
}

class Program
{
    public static Random R = new Random();
    static void Main(string[] args)
    {
        SimpleNet net = new SimpleNet();
        net.Trainingset.Add(new double[] { 0, 0 }, 0);
        net.Trainingset.Add(new double[] { 0, 1 }, 0);
        net.Trainingset.Add(new double[] { 1, 1 }, 1);
        net.Trainingset.Add(new double[] { 1, }, 0);

        Console.WriteLine("Learning...");
        net.Backpropagate(0.1, 0.99);
        Console.WriteLine("{1, 1} " + Treshold(net.Forward(new double[] { 1, 1 }), 0.5)); net.Clear();
        Console.WriteLine("{0, 1} " + Treshold(net.Forward(new double[] { 0, 1 }), 0.5)); net.Clear();
        Console.WriteLine("{1, 0} " + Treshold(net.Forward(new double[] { 1, 0 }), 0.5)); net.Clear();
        Console.WriteLine("{0, 0}" + Treshold(net.Forward(new double[] { 0, 0 }), 0.5)); net.Clear();
        Console.Read();
    }
    static int Treshold(double input, double treshold)
    {
        return input > treshold ? 1 : 0;
    }
}
