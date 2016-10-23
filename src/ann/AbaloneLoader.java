package ann;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Scanner;
import java.util.regex.Pattern;

public class AbaloneLoader {

    public static String ABALONE_TRAIN = "./data/abalone.data";
    public static int NUM_RINGS = 28;
    public static int NUM_CLASSES = 28;
    public static int NUM_INSTANCES = 4177;
    public static int NUM_FEATURES = 8;

    public String file;

    String[] AttributeNames
            = {
                "Sex", "Length", "Diameter", "Height",
                "Whole Weight", "Shucked Weight", "Viscera Weight",
                "Shell Weight", "Rings"
            };

    public AbaloneLoader(String file) {
        this.file = file;
    }

    public ArrayList<Example> getExampleList() {

        ArrayList<Example> examples = new ArrayList<>();
        try {
            // Open the file
            FileInputStream fstream = new FileInputStream(file);

            // Get the object of DataInputStream
            DataInputStream in = new DataInputStream(fstream);

            BufferedReader br = new BufferedReader(new InputStreamReader(in));
            String strLine;

            int cnt = 0;
         
            //Read File Line By Line                
            while ((strLine = br.readLine()) != null) {
                processLine(strLine, examples);
                cnt++;
                //if (cnt == 480) break;
                       
            }

            //Close the input stream
            in.close();
        } catch (Exception e) {//Catch exception if any
            e.printStackTrace();
            System.err.println("Error: " + e.getMessage());
        }
        return examples;
    }

    private void processLine(String line, ArrayList<Example> examples) {
        Scanner s = new Scanner(line);
        s.useDelimiter(",");
        ArrayList<Double> wv = new ArrayList<>();
        String gender = s.next(Pattern.compile("(M|F|I)"));
       // System.out.println(gender);
        if (gender.trim().equalsIgnoreCase("M")) {
            wv.add(1.0);
        } else if (gender.trim().equalsIgnoreCase("F")) {
            wv.add(2.0);
        } else if (gender.trim().equalsIgnoreCase("I")) {
            wv.add(0.0);
        }

        while (s.hasNext()) {
            Double d = s.nextDouble();
            wv.add(d);
        }

        int cls = (int) Math.floor(wv.remove(wv.size() - 1));

        //age is 1 to 29, so we decrease 1
        
        cls = cls - 1;

        double[] darr = new double[wv.size()];
        for (int i = 0; i < darr.length; i++) {
            darr[i] = wv.get(i);
        }

        //if (cls <= NUM_RINGS)
        examples.add(new Example(darr, cls, NUM_RINGS));
    }

    public ArrayList<Example> getCompleteSubset(int subestSize) {
        ArrayList<Example> al = getExampleList();
        ArrayList<Example> subset = new ArrayList<>();
        HashMap<Integer, ArrayList<Example>> map = new HashMap<>();

        for (int i = 0; i < NUM_RINGS; i++) {
            map.put(i, new ArrayList<>());
        }

        for (Example ex : al) {
            try {
                map.get((Integer) (ex.getLabel())).add(ex);

            } catch (Exception e) {
                e.printStackTrace();
            }
        }
       
        int numExamplesPerClass = subestSize/NUM_RINGS;
        for (Integer key : map.keySet()) {
            ArrayList<Example> keyExamples = map.get((Integer) key);
           // System.out.println(keyExamples);
            for (int i = 0; i < numExamplesPerClass; i++) {
                try {
                    subset.add(keyExamples.get(i));
                } catch (Exception e) {
                }
                
            }
        }
        return subset;
    }
}
