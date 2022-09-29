package src.createPopupmenu;
import java.awt.PopupMenu;
import java.awt.MenuItem;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.Color;
import javax.swing.JLabel;
import javax.swing.JPanel;

import src.circlelink.circlelink;

import javax.swing.JColorChooser;

public class createPopupmenu extends PopupMenu{
    public static MenuItem mnext=new MenuItem("Next message");
    public static MenuItem mlib=new MenuItem("Library");
    public static MenuItem mcolor=new MenuItem("Color");
    public static MenuItem mexit=new MenuItem("Exit");
    public createPopupmenu(){
        this.add(mnext);
        this.add(mlib);
        this.add(mcolor);
        this.add(mexit);
    }
    public static void setNext(circlelink clink,JLabel label){
        mnext.addActionListener(new ActionListener(){//下一个
            circlelink clink2=clink;
            public void actionPerformed(ActionEvent e){
                while(clink2.getFlag()==clink2.headrail){//跳过头尾节点
                    clink2=clink2.getNext();
                }
                label.setText(clink2.getData());
            }
        });
    }
    public static void setColorchooser(JPanel panel,JLabel label){
        mcolor.addActionListener(new ActionListener(){//颜色
            public void actionPerformed(ActionEvent e){
                Color color=JColorChooser.showDialog(panel,"选取颜色", null);
                label.setForeground(color);
            }
        });
    }
    public static void setExit(){
        mexit.addActionListener(new ActionListener(){//退出
            public void actionPerformed(ActionEvent e){
                System.exit(0);
            }
        });
    }
}
