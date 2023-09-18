/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package main.Chapter2.mxnet;

import org.apache.mxnet.javaapi.Context;
import org.apache.mxnet.javaapi.DType;
import org.apache.mxnet.javaapi.NDArray;

/**
 *
 * @author omar
 */
public class AutomaticDifferentiation {
  
  public static void main(String[] args) {
  
    NDArray x = NDArray.arange(0, 4, 1, 1, Context.cpu(), DType.Float64());
   
  }
  
}
