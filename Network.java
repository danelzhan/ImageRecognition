import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Network {

    public List<Neuron> layer0;
    public List<Neuron> layer1;
    public List<Neuron> layer2;
    public List<Neuron> layer3;

    public List<Double> expectedOutput;

    public int n0;
    public int n1;
    public int n2;
    public int n3;

    // default untrained network for now
    public Network(int n0, int n1, int n2, int n3) {

        layer0 = new ArrayList<Neuron>();
        layer1 = new ArrayList<Neuron>();
        layer2 = new ArrayList<Neuron>();
        layer3 = new ArrayList<Neuron>();
        expectedOutput = new ArrayList<Double>();

        createLayer(layer0, n0);
        createLayer(layer1, n1);
        createLayer(layer2, n2);
        createOutputLayer(layer3, n3);

        linkLayers(layer0, layer1);
        linkLayers(layer1, layer2);
        linkLayers(layer2, layer3);

        this.n0 = n0;
        this.n1 = n1;
        this.n2 = n2;
        this.n3 = n3;

    }

    public void run() {
        
        for (Neuron n : layer1) {
            n.updateValue();
        }
        for (Neuron n : layer2) {
            n.updateValue();
        }
        for (Neuron n : layer3) {
            n.updateValue();
        }

        applySoftmax(layer3);
        
    }

    void applySoftmax(List<Neuron> outputLayer) {
        double max = Double.NEGATIVE_INFINITY;
        for (Neuron n : outputLayer) {
            max = Math.max(max, n.value);
        }

        double sum = 0.0;
        for (Neuron n : outputLayer) {
            n.value = (float)Math.exp(n.value - max);
            sum += n.value;
        }

        for (Neuron n : outputLayer) {
            n.value /= sum;
        }
    }


    public float getCost() {

        float cost = 0;
        int expected = 0;

        for (int i = 0; i < expectedOutput.size(); i++) {

            double a = layer3.get(i).getValue();
            double y = expectedOutput.get(i);
            
            cost += -(y * Math.log(a) + (1 - y) * Math.log(1 - a));

        }

        return cost / 10;

    }

    public void learn(double stepCoefficient) {

        List<Double> gradient = new ArrayList<Double>();

        getGradient(gradient);
        for (int i = 0; i < n3; i++) {
            for (int j = 0; j < n2; j++) {
                Edge edge = getEdge(j, i, layer3);
                double currentWeight = edge.getWeight();
                //System.out.println("old weight: " + currentWeight);
                edge.setWeight(currentWeight - stepCoefficient * gradient.get(i * n2 + j));
                //System.out.println("new weight: " + edge.getWeight());
                // if (currentWeight != edge.getWeight()) {
                //     System.out.println("changed");
                // }
            }
        }

        for (int j = 0; j < n2; j++) {
            for (int k = 0; k < n1; k++) {
                Edge edge = getEdge(k, j, layer2);
                double currentWeight = edge.getWeight();
                //System.out.println("old weight: " + currentWeight);
                edge.setWeight(currentWeight - stepCoefficient * gradient.get(j * n1 + k));
                //System.out.println("new weight: " + edge.getWeight());
                // if (currentWeight != edge.getWeight()) {
                //     System.out.println("changed");
                // }
            }
        }

        for (int k = 0; k < n1; k++) {
            for (int m = 0; m < n0; m++) {
                Edge edge = getEdge(m, k, layer1);
                double currentWeight = edge.getWeight();
                //System.out.println("old weight: " + currentWeight);
                edge.setWeight(currentWeight - stepCoefficient * gradient.get(k * n0 + m));
                //System.out.println("new weight: " + edge.getWeight());
                // if (currentWeight != edge.getWeight()) {
                //     System.out.println("changed");
                // }
            }
        }

    }

    public void getGradient(List<Double> gradient) {

        // get layer 3 weight gradients
        

        getLayer3WeightGradient(gradient);
        
        getLayer2WeightGradient(gradient);
        getLayer1WeightGradient(gradient);

    }

    public void getLayer3WeightGradient(List<Double> gradient) {

        for (int i = 0; i < n3; i++) {
            for (int j = 0; j < n2; j++) {
                gradient.add(getLayer3Partial(j, i));
                
            }
        }

    }

    public void getLayer2WeightGradient(List<Double> gradient) {

        for (int j = 0; j < n2; j++) {
            for (int k = 0; k < n1; k++) {
                gradient.add(getLayer2Partial(k, j));
                
            }
        }

    }

    public void getLayer1WeightGradient(List<Double> gradient) {

        for (int k = 0; k < n1; k++) {
            for (int m = 0; m < n0; m++) {
                gradient.add(getLayer1Partial(m, k));
            }
        }

    }

    public double getLayer3Partial(int preNeuronIndex, int postNeuronIndex) {

        double postNeuronValue = layer3.get(postNeuronIndex).getValue();
        double preNeuronValue = layer2.get(preNeuronIndex).getValue();

        

        return  -2 * (expectedOutput.get(postNeuronIndex) - postNeuronValue) * 
                // (postNeuronValue * (1 - postNeuronValue)) * 
                preNeuronValue;

    }

    public double getLayer2Partial(int preNeuronIndex, int postNeuronIndex) {

        double postNeuronValue = layer2.get(postNeuronIndex).getValue();
        double preNeuronValue = layer1.get(preNeuronIndex).getValue();

        double result = 0;
        

        for (int i = 0; i < n3; i++) {

            // accounts for derivative of ReLU
            if (postNeuronValue <= 0) continue;

            double L3i = layer3.get(i).getValue();

            Edge edge = getEdge(postNeuronIndex, i, layer3);


            result +=   2 * (expectedOutput.get(i) - L3i) *
                        // (L3i * (1 - L3i)) * 
                        edge.weight * preNeuronValue;

        }

        return -1 * result;

    }

    public double getLayer1Partial(int preNeuronIndex, int postNeuronIndex) {

        double postNeuronValue = layer1.get(postNeuronIndex).getValue();
        double preNeuronValue = layer0.get(preNeuronIndex).getValue();

        double result = 0;

        for (int i = 0; i < n3; i++) {

            double holder = 1;
            double L3i = layer3.get(i).getValue();

            // holder *= 2 * (expectedOutput.get(i) - L3i) * (L3i * (1 - L3i));
            holder *= 2 * (expectedOutput.get(i) - L3i);

            double innerSum = 0;

            for (int j = 0; j < n2; j++) {

                if (layer2.get(j).getValue() <= 0 || postNeuronValue <= 0) continue;
                Edge L3Edge = getEdge(j, i, layer3);
                Edge L2Edge = getEdge(postNeuronIndex, j, layer2);

                innerSum += L3Edge.weight * L2Edge.weight * preNeuronValue;

            }

            holder *= innerSum;

            result += holder;

        }

        return -1 * result;

    }

    public Edge getEdge(int preNeuronIndex, int postNeuronIndex, List<Neuron> layer) {

        Neuron postNeuron = layer.get(postNeuronIndex);
        return postNeuron.getEdge(preNeuronIndex);

    }

    public void createLayer(List<Neuron> layer, int n) {
        for (int i = 0; i < n; i++) {
            Neuron neuron = new Neuron(new ArrayList<Edge>());
            layer.add(neuron);
        }
    }

    public void createOutputLayer(List<Neuron> layer, int n) {
        for (int i = 0; i < n; i++) {
            Neuron neuron = new Neuron(new ArrayList<Edge>());
            neuron.isOutput = true;
            layer.add(neuron);
        }
    }

    public void linkLayers(List<Neuron> preLayer, List<Neuron> postLayer) {

        for (int i = 0; i < postLayer.size(); i++) {

            Neuron postNeuron = postLayer.get(i);

            for (int j = 0; j < preLayer.size(); j++) {

                Neuron preNeuron = preLayer.get(j);
                
                Edge edge = new Edge(preNeuron, postNeuron);
                postNeuron.inputEdges.add(edge);

            }
        }

    }

    public void initWeights() {
        for (Neuron n : layer1) {
            for (Edge e : n.inputEdges) {
                e.initWeight();
            }
        }
        for (Neuron n : layer2) {
            for (Edge e : n.inputEdges) {
                e.initWeight();
            }
        }
        for (Neuron n : layer3) {
            for (Edge e : n.inputEdges) {
                e.initWeight();
            }
        }
    }
    
    public boolean inputValues(double[] data) {
        
        if (data.length != layer0.size()) {
            return false;
        }

        for (int i = 0; i < data.length; i++) {
            layer0.get(i).setValue(data[i]);
        }

        return true;

    }

    public boolean inputExpected(double[] data) {

        expectedOutput = new ArrayList<Double>();

        for (int i = 0; i < data.length; i++) {
            expectedOutput.add(i, data[i]);
        }

        return true;

    }

}
