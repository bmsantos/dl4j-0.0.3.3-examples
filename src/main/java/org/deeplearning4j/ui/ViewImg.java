package org.deeplearning4j.ui;

import com.sun.org.apache.xerces.internal.impl.dv.util.Base64;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;

/**
 * Created by merlin on 7/4/15.
 */
public class ViewImg {

    public void main(String[] args) throws IOException{
        String data = new File("/Users/merlin/Documents/skymind/data/train-images-idx3-ubyte").toString();

        byte[] bytearray = Base64.decode(data);

        BufferedImage imag= ImageIO.read(new ByteArrayInputStream(bytearray));
        File imageFile = new File(System.getProperty("java.io.tmpdir"),"sample.jpg");
        ImageIO.write(imag, "jpg", imageFile);

        System.out.println(imageFile.getPath());

    }

}
