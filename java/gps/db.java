import java.sql.*;
public class db {
    public static Connection getConnection() {
		String driver = "com.mysql.cj.jdbc.Driver";
		String url = "jdbc:mysql://127.0.0.1:3306/gps?characterEncoding=UTF-8&serverTimezone=GMT%2B8";
		String username = "guest";
		String password = "123456";
		Connection conn = null;
        try {
            Class.forName(driver);
			conn = DriverManager.getConnection(url,username,password);
        } catch (Exception e) {
            e.printStackTrace();
        }
		
		return conn;
	}

    public static boolean add() {
        String IP=test.getlocation();
		boolean state = false;
		Connection conn = getConnection();
		if(conn==null) 
			return state;
		PreparedStatement stmt=null;
		String sql = "insert into gps(content) values(?)";
		try {
			stmt = conn.prepareStatement(sql);
			stmt.setString(1,IP);
			stmt.executeUpdate();
			state = true;
			stmt.close();
			conn.close();
		} catch (SQLException e) {
			e.printStackTrace();
		}			
		return state;
	}
    public static void main(String[] args) {
        if(add())
            System.out.println("locate succeed");
        else
            System.out.println("locate failed");
    }
}
