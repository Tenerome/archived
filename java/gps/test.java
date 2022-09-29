import org.jsoup.Connection;
import org.jsoup.Jsoup;
import org.jsoup.select.Elements;
import net.sf.json.JSONException;
import net.sf.json.JSONObject;
import java.io.IOException;
import java.io.Reader;
import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URL;
import java.nio.charset.Charset;
import org.jsoup.nodes.Document;

public class test{
    public static String getPublicIP() {
        String ip = "";
        try {
            Document doc = Jsoup.connect("https://ipchaxun.com/").ignoreContentType(false).get();
            Elements els = doc.select("center");
            for (org.jsoup.nodes.Element el : els) { 
                    ip = el.text(); 
                }
            ip = ip.replaceAll("[^0-9.]", ""); 
            } catch (IOException e1) {
                e1.printStackTrace();
            }finally {
    
            }
            return ip; 
        }

    private static String readAll(Reader rd) throws IOException {
        StringBuilder sb = new StringBuilder();
        int cp;
        while ((cp = rd.read()) != -1) {
            sb.append((char) cp);
        }
        return sb.toString();
    }

    public static JSONObject readJsonFromUrl(String url) throws IOException, JSONException {
        InputStream is = null;
        try {
            is = new URL(url).openStream();
            BufferedReader rd = new BufferedReader(new InputStreamReader(is, Charset.forName("UTF-8")));
            String jsonText = readAll(rd);
            JSONObject json = JSONObject.fromObject(jsonText);
            return json;
        } finally {
            is.close();
        }
    }

    public static String getAddrName(String IP) throws JSONException, IOException{
        JSONObject json = readJsonFromUrl("http://api.map.baidu.com/location/ip?ak=ojrHcoWjiGDZFghONmuIjENitQOwIP4j&ip="+IP);
        String status = json.opt("status").toString();
        if(!"0".equals(status)){
            return "";
        }
        JSONObject content=((JSONObject) json).getJSONObject("content");              
        JSONObject addr_detail=((JSONObject) content).getJSONObject("address_detail");
        String city=addr_detail.opt("province").toString();                            
        city+=addr_detail.opt("city").toString();
        return city;
    }
    public static String getlocation(){
        String IP=getPublicIP();
        String scity="null";
        try {
            scity=getAddrName(IP);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return scity;
    }
}