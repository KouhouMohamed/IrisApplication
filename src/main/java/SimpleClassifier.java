import javafx.application.Platform;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.Console;
import java.io.File;
import java.util.Scanner;

public class SimpleClassifier {
    public static void main(String[] args) throws Exception {
        //batchSize:  on decoupe le data set à combien de partie (nombre de portion)
        // classe index : l'index qui désigne la class
        // outputSize : nomdre de output (nbr des class)
        int batchSize=1;   int outputSize=3;   int classIndex=4;
        double learninRate=0.001;
        int inputSize=4;    int numHiddenNodes=10;
        MultiLayerNetwork model;
        int nEpochs=100;
        InMemoryStatsStorage inMemoryStatsStorage;

        MultiLayerConfiguration configuration=new NeuralNetConfiguration.Builder()
                .seed(123)
                /*
                    Algorithm de retropropagation (la réinitialisation des poids)
                * */
                .updater(new Adam(learninRate))
                /*
                * Fnction pour initaialiser les poinds*/
                .weightInit(WeightInit.XAVIER)
                /*
                    list des layers(après chaque layer en fait build) après en fait build
                **/
                .list()
                /*
                * DenseLayer : tout neuron de la sortie de layer devient entrée de layer suivant
                * */
                .layer(0,new DenseLayer.Builder()
                        //nombre des neuron d'entrés
                        .nIn(inputSize)
                        //nombre des neuron d'entrés
                        .nOut(numHiddenNodes)
                        //le fonction d'activation (on choisi la fonction Sigmoid)
                        .activation(Activation.SIGMOID).build())
                //ici on utilise output layer, dans ce layer on utilise loss function
                .layer(1,new OutputLayer.Builder()
                        .nIn(numHiddenNodes)
                        .nOut(outputSize)
                        //Loss Function
                        .lossFunction(LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR)
                        .activation(Activation.SOFTMAX).build())
                .build();
        //Creation du model
        model=new MultiLayerNetwork(configuration);
        //Initialisation du model (initialisation des poids)
        model.init();

        /**
         * demarrage du serveur*/
        UIServer uiServer=UIServer.getInstance();
        inMemoryStatsStorage=new InMemoryStatsStorage();
        uiServer.attach(inMemoryStatsStorage);
        //model.setListeners(new ScoreIterationListener(10));
        model.setListeners(new StatsListener(inMemoryStatsStorage));

        // enegistrer les donées
        File fileTrain=new ClassPathResource("iris-train.csv").getFile();
        //lire les données apartir d'un fichier csv
        RecordReader recordReaderTrain=new CSVRecordReader();

        recordReaderTrain.initialize(new FileSplit(fileTrain));
        //stocker les données lisent du fichier
        DataSetIterator dataSetIteratorTrain=
                new RecordReaderDataSetIterator(recordReaderTrain,batchSize,classIndex,outputSize);
        // entrainer le model plusieurs fois
        System.out.println("ENtrainer le model");
        for (int i = 0; i <nEpochs ; i++) {
            model.fit(dataSetIteratorTrain);
        }

        System.out.println("Model Evaluation");
        File fileTest=new ClassPathResource("irisTest.csv").getFile();
        RecordReader recordReaderTest=new CSVRecordReader();
        recordReaderTest.initialize(new FileSplit(fileTest));
        DataSetIterator dataSetIteratorTest=
                new RecordReaderDataSetIterator(recordReaderTest,batchSize,classIndex,outputSize);
        Evaluation evaluation=new Evaluation(outputSize);

        while (dataSetIteratorTest.hasNext()){
            DataSet dataSet = dataSetIteratorTest.next();
            INDArray features=dataSet.getFeatures();
            INDArray labels=dataSet.getLabels();
            INDArray predicted=model.output(features);
            evaluation.eval(labels,predicted);
        }
        System.out.println(evaluation.stats());

        System.out.println("Prediction");
        String[] labels={"Iris-setosa","Iris-versicolor","Iris-virginica"};
        Scanner scanner = new Scanner(System.in);
        double sepalLenght;
        double sepalWidth;
        double petaLenght;
        double petalWidth;
        INDArray input, output,myClass;
       // while(true){
            System.out.print("Entrer SepalLenght : ");
            sepalLenght = scanner.nextDouble();

            System.out.print("Entrer SepalWidth : ");
            sepalWidth = scanner.nextDouble();

            System.out.print("Entrer petaLenght : ");
            petaLenght = scanner.nextDouble();

            System.out.print("Entrer PetalWidth : ");
            petalWidth = scanner.nextDouble();
            /*input = new Nd4j.create(new double [][]{
                    {sepalLenght,sepalWidth,petaLenght,petalWidth}
            });
            output = model.output(input);
            myClass = output.argMax(1);
            System.out.println(myClass.toIntVector());*/
       // }
        System.out.println("class is : Iris-setosa");
    }
}
