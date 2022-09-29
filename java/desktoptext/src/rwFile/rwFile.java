package src.rwFile;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

import org.jsoup.select.Evaluator.ContainsData;

import src.circlelink.circlelink;

public class rwFile {
    public static String path;
    public static File file;
    public static FileInputStream input;
    public static byte[] bytes=new byte[1024]; 

    public rwFile(String path,circlelink clink){
        rwFile.path=path;
        initlink(clink);
    }
    public static void initlink(circlelink clink){
        // String path="./lib";路径，相对问题，调用时，路径基于调用的位置而不是当前文件位置，最好把path当做参数传进去
        file=new File(path);
        File[] tempList = file.listFiles();
        for (int i = 0; i < tempList.length; i++) {
            if (tempList[i].isFile()) {
                clink.add(tempList[i].toString());
            }

        }
        for(int i=1;i<=tempList.length;i++){
            file=new File(clink.getI(i).getData());
            try {
                input=new FileInputStream(file);
                if(input.read()==-1){
                    clink.getI(i).setFlag(clink.nocontent);;
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        } 
    }

    public static void writeFile(circlelink clink,String str){//写文件
        
    }

    public static void readFile(circlelink clinkp,circlelink clinkf){//读文件
        for(int i=1;i<=clinkp.getLength();i++){
            if(clinkp.getI(i).getFlag()==clinkp.content){
                file=new File(clinkp.getI(i).getData());
                try {
                    input=new FileInputStream(file);
                    int len=input.read(bytes);
                    String str=new String(bytes,0,len);
                    clinkf.add(str);
                } catch (Exception e) {
                    e.printStackTrace();
                  }
            }

        }
    }
}
