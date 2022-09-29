package src.createFrame;
import javax.swing.JFrame;
import java.awt.Dimension;
import java.awt.Rectangle;
public class createFrame extends JFrame{
    public createFrame(){
        this.setDefaultCloseOperation(3);//点击关闭，3：释放窗口资源
        this.setBounds(new Rectangle(0,0,500,300));
        this.setLocationRelativeTo(null);//相对组件定位，为null时居中 
        this.setResizable(true); 
        this.setMinimumSize(new Dimension(300,150));//设置最小尺寸
        // this.setUndecorated(true);//设置无边框
        // this.setOpacity(0.0f);
        this.setVisible(true);
    }
    public static void main(String[] args) {
        new createFrame();
    }
    
}
