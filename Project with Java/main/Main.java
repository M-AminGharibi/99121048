
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

//import crop.Crop;
import scan.Scan;

public class Main {

    public static void main(String[] args) {

        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        long startTime = System.currentTimeMillis();

        // read image      
        Mat image = Imgcodecs.imread("C:/Users/Amin/Desktop/6.jpg");



        
        /*
        // Automatic cropping 
        Crop autoCrop = new Crop(image, 0.125f, 305);

        Mat croppedImg = autoCrop.cropImage();
        */



        // create an object of Scan class
        Scan scanner = new Scan(image, 51, 66, 160);

        //  "scanImage" method of the Scan class takes the desired mode of operation as input i-e GCMODE/RMODE/SMODE 
            // and then uses a switch-case block to call the appropriate set of functions.
        Mat scannedImg = scanner.scanImage(Scan.ScanMode.RMODE);


        System.out.print(System.currentTimeMillis() - startTime);
        System.out.println(" milliSeconds");

        // write the ouput image as .jpg file
        Imgcodecs.imwrite("C:/Users/Amin/Desktop/scanned.jpg", scannedImg);

    }
    
}