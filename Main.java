import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Arrays;

public class Main {

    public static Network net = new Network(784, 256, 128, 10);
    
    public static void main(String[] args) throws IOException {

        train();

    }

    public static void train() throws IOException {

        Map<String, Integer> trainingData = randomImageMap(1000, 100);
        int epochs = 5;
        net.initWeights();
        for (int i = 0; i < epochs; i++) {

            int count = 0;

            List<String> keys = new ArrayList<String>(trainingData.keySet());

            Collections.shuffle(keys);

            for (String key : keys) {

                String address = key;
                int n = trainingData.get(key);

                double[] expected = new double[10];
                getExpectedResult(n, expected);
                net.inputExpected(expected);
                

                double[] input = new double[784];
                input = imageToBinaryVector(address);
                net.inputValues(input);

                net.run();

                System.out.println("expected: " + n +  " res: " + net.layer3);

                System.out.println("Epoch: " + i + ", Iteration " + count + ", Cost: " + net.getCost());

                net.learn(0.001);

                count++;

            }
        }

        Map<String, Integer> testingData = randomImageMap(500, 30);
        double lossAverage = 0;
        int correctCount = 0;

        for (String key : testingData.keySet()) {

            String address = key;
            int n = testingData.get(key);

            double[] expected = new double[10];
            getExpectedResult(n, expected);
            net.inputExpected(expected);
            

            double[] input = new double[784];
            input = imageToBinaryVector(address);
            net.inputValues(input);

            net.run();

            System.out.println("expected: " + n +  " res: " + net.layer3);
            System.out.println("cost: " + net.getCost());
 
            double greatest = 0;
            double index = 0;
            for (int i = 0; i < net.layer3.size(); i++) {
                if (net.layer3.get(i).getValue() > greatest) {
                    greatest = net.layer3.get(i).getValue();
                    index = i;
                }
            }

            if (index == n) {
                correctCount++;
            }

            double cost = net.getCost();

            if (cost != Double.NaN) {
                lossAverage += net.getCost();
            }

        }

        System.out.println("Average Cost: " + lossAverage / testingData.size());
        System.out.println("Correct: " + correctCount + " Percentage: " + (double)(correctCount)/(testingData.size()));
        
        
    }

    public static void getExpectedResult(int n, double[] expected) {

        for (int i = 0; i < 10; i++) {
            if (i == n)
                {expected[i] = 1.0;} 
            else {
                expected[i] = 0.0;
            }
        }

    }

    public static double[] imageToBinaryVector(String path) throws IOException {
        if (!path.toLowerCase().endsWith(".pgm")) {
            throw new IOException("Only .pgm files are supported: " + path);
        }
        int[][] pgm = readPgm(path);
        int height = pgm.length;
        int width = pgm[0].length;
        int[] pixels = new int[width * height];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                pixels[y * width + x] = pgm[y][x];
            }
        }

        if (width != 28 || height != 28) {
            throw new IOException("Expected 28x28 image, got " + width + "x" + height);
        }

        double[] binary = new double[28 * 28];
        int threshold = 128;
        for (int i = 0; i < binary.length; i++) {
            binary[i] = pixels[i] >= threshold ? 1.0 : 0.0;
        }
        return binary;
    }

    public static HashMap<String, Integer> randomImageMap(
            int totalCount,
            int minPerLabel
    ) throws IOException {
        File baseDir = new File("data/special_train_images");
        if (!baseDir.isDirectory()) {
            throw new IOException("Not a directory: ");
        }

        File[] labelDirs = baseDir.listFiles(File::isDirectory);
        if (labelDirs == null || labelDirs.length == 0) {
            throw new IOException("No label subfolders found in: ");
        }

        Arrays.sort(labelDirs);
        Map<Integer, List<File>> labelFiles = new HashMap<>();
        int totalAvailable = 0;
        for (File labelDir : labelDirs) {
            int label;
            try {
                label = Integer.parseInt(labelDir.getName());
            } catch (NumberFormatException e) {
                continue;
            }
            File[] files = labelDir.listFiles(
                    file -> file.isFile() && file.getName().toLowerCase().endsWith(".pgm")
            );
            if (files == null || files.length == 0) {
                continue;
            }
            List<File> list = new ArrayList<>(Arrays.asList(files));
            labelFiles.put(label, list);
            totalAvailable += list.size();
        }

        if (labelFiles.isEmpty()) {
            throw new IOException("No .pgm files found in: " );
        }

        int labelCount = labelFiles.size();
        int required = Math.max(totalCount, labelCount * minPerLabel);
        if (required > totalAvailable) {
            throw new IOException("Not enough images to satisfy request. " +
                    "Required=" + required + ", available=" + totalAvailable);
        }

        Random rng = new Random();
        HashMap<String, Integer> result = new HashMap<>();
        List<File> remaining = new ArrayList<>();

        for (Map.Entry<Integer, List<File>> entry : labelFiles.entrySet()) {
            List<File> files = entry.getValue();
            Collections.shuffle(files, rng);
            int take = Math.min(minPerLabel, files.size());
            for (int i = 0; i < take; i++) {
                File file = files.get(i);
                result.put(toProjectRelativePath(file), entry.getKey());
            }
            for (int i = take; i < files.size(); i++) {
                remaining.add(files.get(i));
            }
        }

        int remainingNeeded = totalCount - result.size();
        if (remainingNeeded > 0) {
            Collections.shuffle(remaining, rng);
            if (remainingNeeded > remaining.size()) {
                throw new IOException("Not enough remaining images to fill count. " +
                        "Needed=" + remainingNeeded + ", remaining=" + remaining.size());
            }
            for (int i = 0; i < remainingNeeded; i++) {
                File file = remaining.get(i);
                int label = Integer.parseInt(file.getParentFile().getName());
                result.put(toProjectRelativePath(file), label);
            }
        }

        return result;
    }

    private static String toProjectRelativePath(File file) throws IOException {
        Path root = new File(".").getCanonicalFile().toPath();
        Path target = file.getCanonicalFile().toPath();
        try {
            return root.relativize(target).toString();
        } catch (IllegalArgumentException e) {
            return target.toString();
        }
    }

    private static int[][] readPgm(String path) throws IOException {
        try (InputStream input = new BufferedInputStream(new FileInputStream(path))) {
            String magic = nextToken(input);
            if (!"P5".equals(magic) && !"P2".equals(magic)) {
                throw new IOException("Unsupported PGM format: " + magic);
            }
            int width = Integer.parseInt(nextToken(input));
            int height = Integer.parseInt(nextToken(input));
            int maxVal = Integer.parseInt(nextToken(input));
            if (maxVal <= 0 || maxVal > 255) {
                throw new IOException("Unsupported maxVal: " + maxVal);
            }
            int[][] image = new int[height][width];
            if ("P5".equals(magic)) {
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        int value = input.read();
                        if (value < 0) {
                            throw new IOException("Unexpected end of file: " + path);
                        }
                        image[y][x] = value;
                    }
                }
            } else {
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        String token = nextToken(input);
                        if (token == null) {
                            throw new IOException("Unexpected end of file: " + path);
                        }
                        image[y][x] = Integer.parseInt(token);
                    }
                }
            }
            return image;
        }
    }

    private static String nextToken(InputStream input) throws IOException {
        StringBuilder sb = new StringBuilder();
        int c;
        while (true) {
            c = input.read();
            if (c == -1) {
                return null;
            }
            if (c == '#') {
                while (c != -1 && c != '\n' && c != '\r') {
                    c = input.read();
                }
                continue;
            }
            if (!Character.isWhitespace(c)) {
                break;
            }
        }
        sb.append((char) c);
        while (true) {
            c = input.read();
            if (c == -1 || Character.isWhitespace(c) || c == '#') {
                if (c == '#') {
                    while (c != -1 && c != '\n' && c != '\r') {
                        c = input.read();
                    }
                }
                break;
            }
            sb.append((char) c);
        }
        return sb.toString();
    }

}
