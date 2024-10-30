# メモ
- 授業の手法の効率が悪い理由は？
- 完成したら重みが1になっているか確認する。
- 測度を合わせる＝ランダム方向を統一する（なぜかできたからなぜできたか確認）⇒ sampleRandomDirectionでランダムな方向を出して, 拡散ではcomputeDiffuseReflectionの引数に渡して, 
- Pythonで実装した時の乱数はこのレンダラーこ場合は数値ではなく、ランダムな方向を表す。

// 測度の統一
double r1 = randomMT();
double r2 = randomMT();

ここは多分OK

// 拡散反射のPDF
double cos_theta = std::max(0.0, in_n.dot(sampleRandomDirection(in_n, r1, r2).normalized()));
double p_diffuse = cos_theta / M_PI;

ここもOK
なぜなら立体角ベースに統一するから

 // 直接寄与のPDF　1/A*フォンファクタ
 double p_direct = 0.0;
 for (int i = 0; i < in_AreaLights.size(); i++) {
     Eigen::Vector3d n_light = in_AreaLights[i].arm_u.cross(in_AreaLights[i].arm_v);
     double area = n_light.norm() * 4.0;
     if (area > 0.0) {
         p_direct += 1.0 / area;
     }
 }

 ここは関数内で計算しなおす必要なくて、computeDirectLighting関数にPDFを参照渡しして代入されるように変更した方がいい。

 直接光のPDF = 1 / Area * フォンファクタ
 フォンファクタはこの部分
![image](https://github.com/user-attachments/assets/e5bff5b6-72b8-42f9-ab99-78cef18b7321)
ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
これが直接光の被積分関数f(x)
![image](https://github.com/user-attachments/assets/1c7cd554-eb7c-4b21-a21a-e92d22206b77)

Optimal重みを実装するときにはいい感じにfとpを抜き出してくる必要がある。


重みマップを作る！
拡散反射が赤で直接光が青とかのトーンマップを可視化する。








 
