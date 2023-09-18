/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package main.Chapter2.mxnet;

import java.awt.Desktop;
import static java.awt.Desktop.isDesktopSupported;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import org.apache.mxnet.javaapi.Context;
import org.apache.mxnet.javaapi.DType;
import org.apache.mxnet.javaapi.NDArray;
import static java.lang.System.out;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.List;
import java.util.function.Function;
import main.util.PlotlyUtil;
import org.apache.mxnet.javaapi.Shape;
import tech.tablesaw.plotly.Plot;
import tech.tablesaw.plotly.components.Figure;

/**
 *
 * @author omar
 */
public class Calculus {

  public static void main(String[] args) throws IOException, URISyntaxException {
    Function<Double, Double> f = x -> (3 * Math.pow(x, 2) - 4 * x);
    Function<Double, Double> numLimit = x -> ((f.apply(x + 1) - f.apply(1.0)) / x);

    Plot plot = new Plot();
    PlotlyUtil plotlyUtil = new PlotlyUtil();

    Context context = Context.cpu();

    NDArray x = NDArray.arange(-1, -6, -1, 1, context, DType.Float64());

    double[] d = new double[x.size()];

    for (int i = 0; i < d.length; i++) {
      d[i] = Math.pow(10, x.toFloat64Array()[i]);
    }

    for (int i = 0; i < 5; i++) {
      out.printf("h: %.5f", d[i]);
      out.printf(" numerical limit: %.5f %n", numLimit.apply(d[i]));
    }

    //Visualization Utilities
    //args:(start value , end value , step ,repeat , context, and type) 
    x = NDArray.arange(0, 3, (float) 0.1, 1, context, DType.Float64());

    double[] fx = new double[x.size()];
    for (int i = 0; i < x.size(); i++) {
      fx[i] = f.apply(x.toFloat64Array()[i]);
    }

    double[] fg = new double[x.size()];
    for (int i = 0; i < x.size(); i++) {
      fg[i] = 2 * x.toFloat64Array()[i] - 3;
    }

    Figure figure = plotlyUtil.plotLineAndSegment(
        x.toFloat64Array(), fx, fg, "f(x)", "Tangent line(x=1)", "x", "f(x)", 1000, 800);
    //On ubuntu without genome dont forget to install "desktop-file-utils" and run the command. 
    //sudo apt install desktop-file-utils
    //update-desktop-database
    Plot.show(figure);

  }

}
