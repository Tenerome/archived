package entity;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.List;
import javax.servlet.http.HttpServletRequest;

public class major {
    private String id,mname,college;
    
    public major(String id, String mname, String college) {
        super();
        this.id = id;
        this.mname = mname;
        this.college = college;
    }

    public major(){
        this("","","");
    }
    public major(HttpServletRequest request){
        this.id = "";
		this.mname = request.getParameter("mname")==null?"":request.getParameter("mname");
		this.college = request.getParameter("college")==null?"":request.getParameter("college");
    }

    public static List<major> getMajors(){ //获取所有专业信息            test -ok
		List<major> majors = new ArrayList<major>();
		Connection conn = db.getConnection();
		ResultSet rs = null;
		if(conn==null) 
			return majors;
		Statement stmt=null;		
	    String sql = "select * from major";	    
		try {
			stmt = conn.createStatement();
			rs = stmt.executeQuery(sql);
			while(rs.next()) {				
				majors.add(new major(rs.getString("id"),//int型可以直接getstring
						             rs.getString("mname"),
						             rs.getString("college")));
			}			
			rs.close();
			stmt.close();
			conn.close();
		} catch (SQLException e) {
			e.printStackTrace();
		}	
		
		return majors;
	}

    public static String getMajorforSelect() {
    	String html="";
    	List<major> majors = getMajors();
    	for(int i=0;i<majors.size();i++) {
    		html+="<option value="+majors.get(i).getId()+">"+majors.get(i).getMname()+"</option>";
    	}
    	
    	return html;		
	}
    
    public boolean add() {
    	boolean state = false;
		Connection conn = db.getConnection();
		if(conn==null) 
			return state;
		PreparedStatement stmt=null;
		String sql = "insert into major(mname,college) values(?,?)";
		try {
			stmt = conn.prepareStatement(sql);
			stmt.setString(1, this.getMname());
			stmt.setString(2, this.getCollege());			
			stmt.executeUpdate();
			state = true;
			stmt.close();
			conn.close();
		} catch (SQLException e) {
			e.printStackTrace();
		}			
		return state;
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getMname() {
        return mname;
    }

    public void setMname(String mname) {
        this.mname = mname;
    }

    public String getCollege() {
        return college;
    }

    public void setCollege(String college) {
        this.college = college;
    }
    //for test
    // public static void main(String[] args) {
    //     System.out.println(major.getMajors().get(1).getMname());
        
    // }
}
