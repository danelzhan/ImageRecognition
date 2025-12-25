import java.util.List;

public class Neuron {

    public List<Edge> inputEdges;
    public float value;
    public float bias;
    public boolean isOutput = false;

    public Neuron(List<Edge> inputEdges) {
        this.inputEdges = inputEdges;
        this.value = 0;
        this.bias = 0;
    }

    public Neuron(List<Edge> inputEdges, float value, float bias) {
        this.inputEdges = inputEdges;
        this.value = value;
        this.bias = bias;
    }

    public boolean addEdge(Edge e) {

        // implement error checking later
        this.inputEdges.add(e);
        return true;

    }

    public Edge getEdge(int preNeuronIndex) {

        // since when creating network edges, preNeuronIndex matches with index of edge in inputEdges
        return inputEdges.get(preNeuronIndex);

    }

    public void setValue(double value) {
        this.value = (float)value;
    }

    public float getValue() {
        return this.value;
    }

    public void updateValue() {

        float sum = 0;
        
        
        for (int i = 0; i < inputEdges.size(); i++) {
            Edge e = inputEdges.get(i);
            // System.out.println("test" + e.weight);
            // if (sum != 0) System.out.println(sum);
            sum += e.weight * e.preNeuron.value;
            
            
        }

        
        
        sum += this.bias;

        if (isOutput) {
            this.value = sum;

        } else {
            this.value = ReLU(sum);
        }

    }

    private float ReLU(float x) {
        return Math.max(0, x);
    }

    private float sigmoid(float x) {
        return (float)(1.0 / (1.0 + Math.exp(-x)));
    }

    @Override
    public String toString() {
        // return "{Value: " + value + ", Bias: " + bias + ", Inputs: " + inputEdges.size() + "}";
        return "" + value;
    }

    
}
