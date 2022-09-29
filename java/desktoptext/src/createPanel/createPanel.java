package src.createPanel;
import javax.swing.JPanel;
import javax.swing.JFrame;

public class createPanel extends JPanel {
    public createPanel(){
        this.setOpaque(true);
    }    
    public createPanel(JFrame frame){
        this.setSize(frame.getWidth(),frame.getHeight());
    }
}
