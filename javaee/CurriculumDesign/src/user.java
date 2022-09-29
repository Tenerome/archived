package entity;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import javax.servlet.http.HttpServletRequest;

public class user {
    private String username, password;
    public user(String username,String password){
        super();
        this.username=username;
        this.password=password;
    }
    public user(){
        this("","");
    }
    public user(HttpServletRequest request) {	
        super();
		this.username = request.getParameter("username")==null?"":request.getParameter("username");
		this.password = request.getParameter("password")==null?"":request.getParameter("password");
	}

    public boolean check(){//验证  ok
        boolean state = false;
		Connection conn = db.getConnection();
		ResultSet rs = null;
		if(conn==null) 
			return state;
		PreparedStatement stmt=null;
	    String sql = "select * from user where username=? and password=?";
	    try {
			stmt = conn.prepareStatement(sql);
			stmt.setString(1, this.getUsername());
			stmt.setString(2, this.getPassword());
			rs = stmt.executeQuery();
			if(rs.next())
                state=true;	
		} catch (SQLException e) {
			e.printStackTrace();
		}
	    return state;
    }
    public boolean register(){//验证  ok
        boolean state = false;
		Connection conn = db.getConnection();
		ResultSet rs = null;
		if(conn==null) 
			return state;
		PreparedStatement stmt=null;
	    String sql = "insert into user(username,password) values(?,?)";
	    try {
			stmt = conn.prepareStatement(sql);
			stmt.setString(1, this.getUsername());
			stmt.setString(2, this.getPassword());
			stmt.executeUpdate();
            state=true;	
			stmt.close();
			conn.close();
		
		} catch (SQLException e) {
			e.printStackTrace();
		}
	    return state;
    }

    public String getUsername() {
        return username;
    }

    public void setUsername(String username) {
        this.username = username;
    }

    public String getPassword() {
        return password;
    }

    public void setPassword(String password) {
        this.password = password;
    }
//for test
// public static void main(String[] args) {
//         user uu=new user("guest","123456");
//         if(uu.check())
//             System.out.println("success");
// }

}