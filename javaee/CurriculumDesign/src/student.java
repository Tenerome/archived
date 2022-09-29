package entity;

import javax.servlet.http.HttpServletRequest;

public class student {
    private String id,sno, sname, ssex, smajor, sgrade;

    public student(String id, String sno, String sname, String ssex, String smajor, String sgrade) { // ok
        super();
        this.id=id;
        this.sno = sno;
        this.sname = sname;
        this.ssex = ssex;
        this.smajor = smajor;
        this.sgrade = sgrade;
    }
    public student(String sno, String sname, String ssex, String smajor, String sgrade){
        this("",sno,sname,ssex,smajor,sgrade);
    }
 

    public student(HttpServletRequest request) {// to do test
        super();
        this.id=request.getParameter("id") == null ? "" : request.getParameter("id");
        this.sno = request.getParameter("sno") == null ? "" : request.getParameter("sno");
        this.sname = request.getParameter("sname") == null ? "" : request.getParameter("sname");
        this.ssex = request.getParameter("ssex") == null ? "" : request.getParameter("ssex");
        this.smajor = request.getParameter("smajor") == null ? "" : request.getParameter("smajor");
        this.sgrade = request.getParameter("sgrade") == null ? "" : request.getParameter("sgrade");
    }

    public student() { // ok
        this("","", "", "", "", "");
    }

    public boolean isAllEmpty(){
        if(this.sno.equals("")&&this.sname.equals("")&&this.ssex.equals("")&&this.smajor.equals("")&&this.sgrade.equals(""))
            return true;
        else
            return false;
    }
    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }
    public String getSno() {
        return sno;
    }

    public void setSno(String sno) {
        this.sno = sno;
    }

    public String getSname() {
        return sname;
    }

    public void setSname(String sname) {
        this.sname = sname;
    }

    public String getSsex() {
        return ssex;
    }

    public void setSsex(String ssex) {
        this.ssex = ssex;
    }

    public String getSmajor() {
        return smajor;
    }

    public void setSmajor(String smajor) {
        this.smajor = smajor;
    }

    public String getSgrade() {
        return sgrade;
    }

    public void setSgrade(String sgrade) {
        this.sgrade = sgrade;
    }

    // // test
    // public static void main(String[] args) {
    //     student s1 = new student();
    //     s1.setSgrade("100");

    // }

}
