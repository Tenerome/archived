package entity;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.List;

public class db {
	public static Connection getConnection(String driver, String url, String username, String password) { // 链接数据库方法
		Connection conn = null;
		try {
			Class.forName(driver);
			conn = DriverManager.getConnection(url, username, password);
		} catch (ClassNotFoundException e) {
			System.out.println("数据库驱动加载失败");
			e.printStackTrace();
		} catch (SQLException e) {
			System.out.println("数据库连接失败");
			e.printStackTrace();
		}
		return conn;

	}

	public static Connection getConnection() { // 重载
		String driver = "com.mysql.cj.jdbc.Driver";
		String url = "jdbc:mysql://127.0.0.1:3306/test?characterEncoding=UTF-8&serverTimezone=GMT%2B8";
		String username = "guest";
		String password = "123456";
		Connection conn = null;
		conn = getConnection(driver, url, username, password);
		return conn;
	}

	public static List<student> getStudents(student stu, int page, int record) {// 查询所有学生信息,page,recode存页数和每页记录数
		List<student> students = new ArrayList<student>();
		Connection conn = getConnection();
		ResultSet rs = null;
		if (conn == null)
			return students;
		Statement stmt = null;
		String sql = "select * from student";
		if (!stu.isAllEmpty()) {
			sql += " where ";
			if (!stu.getSno().equals(""))
				sql += "sno like '%" + stu.getSno() + "%' and ";
			if (!stu.getSname().equals(""))
				sql += "sname like '%" + stu.getSname() + "%' and ";
			if (!stu.getSsex().equals(""))
				sql += "ssex like '%" + stu.getSsex() + "%' and ";
			if (!stu.getSmajor().equals(""))
				sql += "smajor like '%" + stu.getSmajor() + "%' and ";
			if (!stu.getSgrade().equals(""))
				sql += "sgrade like '%" + stu.getSgrade() + "%' and ";
			sql = sql.substring(0, sql.length() - 5); // 去掉and
		}
		sql += " order by sno";
		if (page > 0) {
			int begin = (page - 1) * record;
			sql += " limit " + begin + "," + record;// sql分页查询
		}
		try {
			stmt = conn.createStatement();
			rs = stmt.executeQuery(sql);
			while (rs.next()) {				//这里获取数据库的int型id，可以直接getString也可以getInt后用String.valueOf()转换
				students.add(new student(rs.getString("id"),rs.getString("sno"), rs.getString("sname"),
						rs.getString("ssex"),rs.getString("smajor"), rs.getString("sgrade")));//arraylist的add(E object),调用student的构造方法
						                                                                      
			}
			rs.close();
			stmt.close();
			conn.close();
		} catch (SQLException e) {
			
			e.printStackTrace();
		}

		return students;
	}

	public static List<student> getStudents(student stu) {//重载，不分页
		return getStudents(stu, 0, 0);
	}
	
	public static List<student> getStudents() {//重载
		return getStudents(new student(), 0, 0);
	}

	public static boolean add(student stu) {//添加
		boolean state = false;
		Connection conn = getConnection();
		if(conn==null) 
			return state;
		PreparedStatement stmt=null;
		String sql = "insert into student(sno,sname,ssex,smajor,sgrade) values(?,?,?,?,?)";
		try {
			stmt = conn.prepareStatement(sql);
			stmt.setString(1, stu.getSno());
			stmt.setString(2, stu.getSname());
			stmt.setString(3, stu.getSsex());
			stmt.setString(4, stu.getSmajor());
			stmt.setString(5, stu.getSgrade());
			stmt.executeUpdate();
			state = true;
			stmt.close();
			conn.close();
		} catch (SQLException e) {
			e.printStackTrace();
		}			
		return state;
	}
	
	public static boolean update(student stu) {  //更新
		boolean state = false;
		Connection conn = getConnection();
		if(conn==null) 
			return state;
		PreparedStatement stmt=null;
		String sql = "update student set sno=?, sname=?,ssex=?,smajor=?,sgrade=? where id=?";
		try {
			stmt = conn.prepareStatement(sql);
			stmt.setString(1, stu.getSno());
			stmt.setString(2, stu.getSname());
			stmt.setString(3, stu.getSsex());
			stmt.setString(4, stu.getSmajor());
			stmt.setString(5, stu.getSgrade());
			stmt.setInt(6, Integer.parseInt(stu.getId()));
			stmt.executeUpdate();
			state = true;
			stmt.close();
			conn.close();
		} catch (SQLException e) {
			e.printStackTrace();
		}			
		return state;
	}
	
	public static boolean delete(String id) {//删除
		boolean state = false;
		Connection conn = getConnection();
		if(conn==null) 
			return state;
		PreparedStatement stmt=null;
		String sql = "delete from student where id=?";
		try {
			stmt = conn.prepareStatement(sql);
			stmt.setInt(1, Integer.parseInt(id));			
			stmt.executeUpdate();
			state = true;
			stmt.close();
			conn.close();
		} catch (SQLException e) {
			e.printStackTrace();
		}
		return state;
	}
	public static String getStuHTML(student stu,int page,int record) {//返回html表格
		String html="";
		List<student> students = getStudents(stu,page,record);
		html+="<div id='table'>\n";
		html+="<table>\n";
		html+="<tr>";
		html+="<th>序号</th>";
		html+="<th>学号</th>";
		html+="<th>姓名</th>";
		html+="<th>性别</th>";
		html+="<th>专业</th>";
		html+="<th>成绩</th>";
		html+="<th>操作</th>";
		html+="<th></th>";
		html+="</tr>\n";
		html+="</table>\n";
		for(int i=0;i<students.size();i++) {
			html+="<table>\n";
			html+="<tr>";
			html+="<td>"+students.get(i).getId()+"</td>";
			html+="<td>"+students.get(i).getSno()+"</td>";
			html+="<td>"+students.get(i).getSname()+"</td>";
			html+="<td>"+students.get(i).getSsex()+"</td>";
			html+="<td>"+students.get(i).getSmajor()+"</td>";
			html+="<td>"+students.get(i).getSgrade()+"</td>";
			html+="<td><a href='index.jsp?operate=update.jsp&id="+students.get(i).getId()+"'>修改</a>&nbsp;&nbsp;<a href='delete?id="+students.get(i).getId()+"'>删除</a></td>";
			html+="</tr>\n";
			html+="</table>\n";
		}
		// html+="<div>\n";
		//html+="<a href=''>第一页</a>";	
		//html+="<a href=''>上一页</a>";	
		//html+="<a href=''>下一页</a>";	
		//html+="<a href=''>最后一页</a>";	
		html+="</div>\n";
		
		return html;
	}
	public static student getStudentByID(String id) {		//test-ok
		Connection conn = getConnection();
		student stu = new student();
		ResultSet rs = null;
		if(conn==null) 
			return stu;
		PreparedStatement stmt=null;
		String sql = "select * from student where id=?";
		try {
			stmt = conn.prepareStatement(sql);
			stmt.setInt(1, Integer.parseInt(id));
			rs = stmt.executeQuery();
			if(rs.next())
				stu = new student(rs.getString("id"),rs.getString("sno"),rs.getString("sname"),
					rs.getString("ssex"),rs.getString("smajor"),rs.getString("sgrade"));			
		
		} catch (SQLException e) {
			e.printStackTrace();
		}
		return stu;
	}
	// for test
	// public static void main(String[] args) {
	// 	db.getConnection();
	// 	student stu=new student("2","21231","达瓦","男","计算机","23");
	// 	System.out.println(db.update(stu));
	//  }
}