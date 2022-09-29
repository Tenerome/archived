package src.circlelink;

public class circlelink {
    private String data;//文本信息
    private int flag;//标识文件内容,-1：头尾节点，0：无内容，1：有内容
    private circlelink next;//下一个节点
    private int length=0;//长度
    private circlelink last;//尾节点
    public final int headrail=-1,nocontent=0,content=1;//标识符
    

    public circlelink(){//构造
        super();
        this.data="头";
        this.flag=headrail;
    };

    public void init(){//初始化
        last=new circlelink();
        this.next=last;
        last.data="尾";
        last.flag=headrail;
        last.next=this;
    }

    public void add(String str){//添加节点
        circlelink x=this;
        int len=this.length;
        while(len!=0){
            len--;
            x=x.next;
        }
        circlelink y=new circlelink();
        x.next=y;
        y.data=str;
        y.flag=content;
        y.next=last;
        length++;
    };
    public circlelink getI(int i){//获取第i个,从1开始



        circlelink clink=this;
        if(i>clink.length){
            errorI();
            return null;
        }
        while(i!=0){
            clink=clink.next;
            if(clink.flag==content){
                i--;
            }
            
           
        }
        if(clink.flag==headrail){
            errorI();
            return null;
        }else
        return clink;
    }

    public void errorI(){
        System.out.println();
        System.err.println("Position I is not available,please check I");
        System.out.println();
    }
    public void printLink(){//输出所有
        circlelink x=this;
        int len=length;
        while(len!=0){
            if(x.flag==content){
                System.out.println(x.data);
                len--;
            }
            x=x.next;
        }
    }
    public String[] getString(){
        circlelink x=this;
        String[] str=new String[x.length];
        int len=length;
        while(len!=0){
            if(x.flag==content){
                str[len]=x.data;
                len--;
            }
            x=x.next;
        }
        return str;
    }
    //getter and setter
    public void setData(String data){
        this.data=data;
    }

    public String getData(){
        return this.data;
    }
    
    public void setFlag(int flag){
        this.flag=flag;
    }

    public int getFlag(){
        return this.flag;
    }

    public void setNext(circlelink next){
        this.next=next;        
    }

    public circlelink getNext(){
        return this.next;
    }
    public int getLength(){
        return this.length;
    }

    //main
    // public static void main(String[] args) {
    //     circlelink clink=new circlelink();
    //     clink.init();
    //     clink.add("你好");
    //     clink.add("世界");
    //     // clink.printLink();
    //     System.err.println(clink.getI(2).data);
    // }
}

