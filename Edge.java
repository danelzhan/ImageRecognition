public class Edge {

    public double weight;
    public Neuron preNeuron;
    public Neuron postNeuron;

    public Edge(Neuron preNeuron, Neuron postNeuron) {

        this.preNeuron = preNeuron;
        this.postNeuron = postNeuron;

    }

    public Edge(double weight, Neuron preNeuron, Neuron postNeuron) {

        this.weight = weight;
        this.preNeuron = preNeuron;
        this.postNeuron = postNeuron;

    }

    public void initWeight() {
        this.weight =   -Math.sqrt(2.0 / postNeuron.inputEdges.size())
                        + (float)Math.random()
                        * (Math.sqrt(2.0 / postNeuron.inputEdges.size())
                        - -Math.sqrt(2.0 / postNeuron.inputEdges.size()));
    }

    public Neuron getPreNeuron() {
        return preNeuron;
    }

    public void setWeight(double weight) {
        this.weight = weight;
    }

    public double getWeight() {
        return this.weight;
    }

    @Override
    public String toString() {
        return "{Weight: " + weight + "}";
    }
    
}
