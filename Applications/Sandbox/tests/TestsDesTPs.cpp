#include <Unitest/Unitest.h>
#include <Unitest/TestMacro.h>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <iostream>
#include <random>
#include <cstdlib>

#include "NKLogger/NkLog.h"
#include "NKMath/NKMath.h"
#include "Mat4d.h" 
#include "Quat.h" 
#include "NKImage.h" 


using namespace nkentseu::math;

const int width = 512, height = 512;
NkImage img(width, height);

std::mt19937 rng(42);
std::uniform_real_distribution<double> dist(-10.0, 10.0);

std::vector<Vec4d> cube = {
    {-0.5,-0.5,-0.5,1}, {0.5,-0.5,-0.5,1},
    {0.5, 0.5,-0.5,1}, {-0.5, 0.5,-0.5,1},
    {-0.5,-0.5, 0.5,1}, {0.5,-0.5, 0.5,1},
    {0.5, 0.5, 0.5,1}, {-0.5, 0.5, 0.5,1}
};
    
std::vector<Vec2d> edges = {
    {0,1},{1,2},{2,3},{3,0}, // face arrière
    {4,5},{5,6},{6,7},{7,4}, // face avant
    {0,4},{1,5},{2,6},{3,7}  // connexions
};

Vec3d eye{0,1,3}, target{0,0,0}, up{0,1,0};
Mat4d V = LookAt(eye, target, up);
Mat4d P = Perspective(60.0, double(width)/height, 0.1, 100.0);


// TP1 - Semaine 1 : Test de inspectFloat pour analyser les flottants IEEE 754
TEST_CASE(InspectionFlottant, Semaine1_TP1) {
    // Tests des cas spéciaux flottants
    inspectFloat(0.1f);                          // Décimale simple
    inspectFloat(1.0f);                          // Entier positif
    inspectFloat(1.0f / 0.0f);                   // Infini positif
    inspectFloat(std::sqrt(-1.0f));              // NaN
    inspectFloat(-0.0f);                         // Zéro négatif
    inspectFloat(0.0f);                          // Zéro positif
    inspectFloat(std::numeric_limits<float>::min()); // Plus petit flottant positif normalisé
}


// TP2 - Semaine 1 : Problèmes de précision flottante
TEST_CASE(ProblemesPrecision, Semaine1_TP2) {
    float s1, s2;  // Sommes
    std::vector<float> v;  // Pour variance

    // Grand tableau pour erreurs d'arrondi
    std::vector<float> data(1'000'000, 0.1f);

    // Comparaison accumulate vs Kahan
    s1 = std::accumulate(data.begin(), data.end(), 0.0f);
    s2 = kahanSum(data);
    logger.Info("\nSomme accumulate : {0}\nSomme Kahan : {1}\nAttendue : 100000.0f", s1, s2);

    // Variance naïve vs Welford
    v = std::vector<float>({1e8f, 1e8f, 1.0f, 2.0f});
    logger.Info("\nVariance naïve : {0}\nVariance Welford : {1}", varianceNaive(v), varianceWelford(v));

    // Epsilon machine
    logger.Info("\nEpsilon calculé : {0}\nEpsilon std : {1}", epsilonMachine(), std::numeric_limits<float>::epsilon());
}


// TP3 - Semaine 1 : Tests fonctions Float.h (33 tests)
TEST_CASE(TestsFonctionsFlottant, Semaine1_TP3) {
    // isFiniteValid (5 tests)
    ASSERT_TRUE(!isFiniteValid(std::numeric_limits<float>::quiet_NaN()));     // NaN pas fini
    ASSERT_TRUE(!isFiniteValid(std::numeric_limits<float>::infinity()));      // +Inf pas fini
    ASSERT_TRUE(!isFiniteValid(-std::numeric_limits<float>::infinity()));     // -Inf pas fini
    ASSERT_TRUE(isFiniteValid(0.0f));                                         // Zéro fini
    ASSERT_TRUE(isFiniteValid(1.0f));                                         // Nombre fini

    // nearlyZero (8 tests)
    float values[] = {0.0f, 1e-7f, -1e-7f, 1e-5f, -1e-5f, 1e-3f, -1e-3f, 5e-8f};
    float epsilons[] = {1e-6f, 1e-6f, 1e-6f, 1e-2f, 1e-7f, 1e-2f, 1e-3f, 1e-3f};

    for (int i = 0; i < 8; ++i) {
        float x = values[i];
        float eps = epsilons[i];

        if (std::fabs(x) < eps)
            ASSERT_TRUE(nearlyZero(x, eps));  // Proche de zéro
        else
            ASSERT_TRUE(!nearlyZero(x, eps)); // Pas proche de zéro
    }

    // approxEq (11 tests)
    float a_vals[] = {1.0f, 1.0f, 0.0f, 0.0f, -1.0f, -1.0f, 1000.0f, 1000.0f};
    float b_vals[] = {1.0f, 1.00001f, 1e-7f, 1e-3f, -1.00001f, -1.1f, 1000.0001f, 1001.0f};
    float eps_vals[] = {1e-6f, 1e-1f, 1e-3f, 1e-6f, 1e-9f, 1e-4f, 1e-2f, 1e-5f};
    for (int i = 0; i < 8; ++i) {
        float a = a_vals[i];
        float b = b_vals[i];
        float eps = eps_vals[i];

        if (std::fabs(a - b) <= eps)
            ASSERT_TRUE(approxEq(a, b, eps));  // Égaux approximativement
        else
            ASSERT_TRUE(!approxEq(a, b, eps)); // Pas égaux approximativement
    }

    // Tests supplémentaires approxEq
    float small_vals[] = {1e-7f, 2e-7f, 5e-7f};
    for (int i = 0; i < 3; ++i) {
        ASSERT_TRUE(approxEq(0.0f, small_vals[i], 1e-6f));  // Petites valeurs
    }

    // 4. Tests de kahanSum vs accumulate (10 tests) : Comparaison de la précision des sommes
    // Utilise différents tailles de tableaux pour tester la robustesse
    std::vector<int> sizes = {1000, 10000, 50000, 344530, 76654, 999999, 123456, 654321, 1000000};
    for (int n : sizes) {
        std::vector<float> v(n, 0.1f);  // Tableau de n éléments égaux à 0.1f

        float s1 = std::accumulate(v.begin(), v.end(), 0.0f);  // Somme naïve
        float s2 = kahanSum(v);  // Somme compensée

        float expected = n * 0.1f;  // Valeur attendue exacte

        // La somme de Kahan doit être plus proche de la valeur attendue
        ASSERT_TRUE(std::fabs(s2 - expected) < std::fabs(s1 - expected));
    }

    // Tests supplémentaires avec des valeurs spécifiques pour tester les annulations
    std::vector<std::vector<float>> tests = {
        {1e8f, 1.0f, -1e8f},           // Annulation de grande échelle
        {1.0f, 1e8f, -1e8f},           // Ordre différent
        {1e7f, 1.0f, 1.0f, -1e7f},    // Annulation avec échelle moyenne
        {-1e10f, 3e10f, 1.0f, -2e10f, 1e10f, -1e10f}  // Cas complexe
    };
    for (auto& v : tests) {
        float s1 = std::accumulate(v.begin(), v.end(), 0.0f);
        float s2 = kahanSum(v);

        float expected = (v.size() == 4) ? 2.0f : 1.0f;  // Résultat attendu selon la taille
        ASSERT_TRUE(std::fabs(s2 - expected) <= std::fabs(s1 - expected));
    }

    // Tests avec différentes configurations de taille et valeur
    std::vector<std::pair<int, float>> configs = {
        {100000, 0.01f},   // Petit incrément
        {100000, 1e-5f},   // Très petit incrément
        {50000, 0.2f}      // Incrément plus grand
    };
    for (auto& [n, val] : configs) {
        std::vector<float> v(n, val);

        float s2 = kahanSum(v);
        float expected = n * val;
        ASSERT_TRUE(approxEq(s2, expected, 1e-2f));  // Tolérance pour les erreurs d'arrondi
    }
}


// --------------------------------  TP4 - Semaine 2 : Implémentation complète de Vec2d avec 20 tests
// Ce test couvre les opérations fondamentales sur les vecteurs 2D : produit scalaire,
// produit vectoriel 2D, normalisation, accès par index, et vérification de la taille mémoire.
TEST_CASE(ImplementationVec2d, Semaine2_TP1) {
    // 1. Tests du produit scalaire (Dot product) - 6 tests
    // Vérifie les propriétés du produit scalaire avec des vecteurs orthogonaux, parallèles, etc.
    ASSERT_TRUE(Dot({1,0}, {0,1}) == 0.0);      // Vecteurs orthogonaux
    ASSERT_TRUE(Dot({1,0}, {1,0}) == 1.0);      // Vecteurs identiques unitaires
    ASSERT_TRUE(Dot({3,4}, {3,4}) == 25.0);     // Vecteurs identiques (3-4-5)
    ASSERT_TRUE(Dot({-1,0}, {1,0}) == -1.0);    // Vecteurs opposés
    ASSERT_TRUE(Dot({2,3}, {4,5}) == 23.0);     // Calcul général
    ASSERT_TRUE(Dot({0,0}, {5,7}) == 0.0);      // Un vecteur nul

    // 2. Tests du produit vectoriel 2D (Cross2D) - 4 tests
    // Le produit vectoriel 2D donne l'aire du parallélogramme orienté
    ASSERT_TRUE(Cross2D({1,0}, {0,1}) == 1.0);   // Base canonique
    ASSERT_TRUE(Cross2D({0,1}, {1,0}) == -1.0);  // Ordre inversé
    ASSERT_TRUE(Cross2D({1,1}, {1,1}) == 0.0);   // Vecteurs colinéaires
    ASSERT_TRUE(Cross2D({2,0}, {0,2}) == 4.0);   // Calcul général

    // 3. Tests de normalisation - 4 tests
    // La normalisation produit un vecteur unitaire dans la même direction
    Vec2d w = {3,4};
    Vec2d n = w.Normalized();
    ASSERT_TRUE(std::fabs(n.Norm() - 1.0) < kEps);   // Norme doit être 1

    // Direction conservée : (3,4) normalisé donne (0.6, 0.8)
    ASSERT_TRUE(std::fabs(n.x - 0.6) < kEps);    // Composante x
    ASSERT_TRUE(std::fabs(n.y - 0.8) < kEps);    // Composante y

    // Vecteur unitaire reste inchangé
    Vec2d u = {1,0};
    u = u.Normalized();
    ASSERT_TRUE(std::fabs(u.x - 1.0) < kEps);   // Vecteur déjà unitaire

    // 4. Tests de l'opérateur [] - 5 tests
    // Accès aux composantes par index (0 pour x, 1 pour y)
    w = {10, 20};
    ASSERT_TRUE(w[0] == 10.0);   // Accès à x
    ASSERT_TRUE(w[1] == 20.0);   // Accès à y

    w[0] = 30;  // Modification via index
    ASSERT_TRUE(w.x == 30.0);    // Vérification de la modification

    w[1] = 40;
    ASSERT_TRUE(w.y == 40.0);    // Vérification de la modification

    u = {5, 6};
    ASSERT_TRUE(u[0] == 5.0);    // Test avec autre vecteur

    // 5. Vérification de la taille mémoire - 1 test
    // Vec2d doit occuper exactement 16 octets (2 doubles de 8 octets chacun)
    static_assert(sizeof(Vec2d) == 16, "Vec2d doit faire 16 octets");
}

// TP5 - Semaine 2 : Vec3d Gram-Schmidt
TEST_CASE(Vec3dGramSchmidt, Semaine2_TP2) {
    // 1. Tests du produit vectoriel (Cross product) - 6 tests
    // Définition des vecteurs de base
    Vec3d i = {1,0,0}, j = {0,1,0}, k = {0,0,1};

    // Règle de la main droite : i × j = k
    ASSERT_TRUE(ApproxVec(Cross(i, j), k));               // Produit vectoriel de base
    ASSERT_TRUE(ApproxVec(Cross(j, i), {0,0,-1}));        // Ordre inversé donne l'opposé
    // Base complète : j × k = i, k × i = j
    ASSERT_TRUE(ApproxVec(Cross(j, k), i));
    ASSERT_TRUE(ApproxVec(Cross(k, i), j));
    // Orthogonalité : (i × j) · i = 0 et (i × j) · j = 0
    ASSERT_TRUE(approxEq(Dot(Cross(i, j), i), 0));
    ASSERT_TRUE(approxEq(Dot(Cross(i, j), j), 0));

    // 2. Orthogonalisation de Gram-Schmidt sur 10 triplets aléatoires - 10 tests
    // Construction d'une base orthonormale à partir de trois vecteurs
    for(int t = 0; t < 10; ++t) {
        Vec3d a{dist(rng), dist(rng), dist(rng)};
        Vec3d b{dist(rng), dist(rng), dist(rng)};
        Vec3d c{dist(rng), dist(rng), dist(rng)};

        // Processus de Gram-Schmidt
        Vec3d ui = a.Normalized();  // Premier vecteur unitaire
        Vec3d vi = (b - Project(b, ui)).Normalized();  // Orthogonalisation du deuxième
        Vec3d wi = (c - Project(c, ui) - Project(c, vi)).Normalized();  // Orthogonalisation du troisième

        // Vérification des normes (doivent être 1)
        ASSERT_TRUE(approxEq(ui.Norm(), 1.0));
        ASSERT_TRUE(approxEq(vi.Norm(), 1.0));
        ASSERT_TRUE(approxEq(wi.Norm(), 1.0));

        // Vérification de l'orthogonalité : ui · vi = 0, ui · wi = 0, vi · wi = 0
        ASSERT_TRUE(approxEq(Dot(ui, vi), 0.0));
        ASSERT_TRUE(approxEq(Dot(ui, wi), 0.0));
        ASSERT_TRUE(approxEq(Dot(vi, wi), 0.0));
    }

    // 3. Tests de Project et Reject - 10 tests
    // Projection et rejet : proj + rej = original
    for(int t = 0; t < 10; ++t) {
        Vec3d a{dist(rng), dist(rng), dist(rng)};
        Vec3d b{dist(rng), dist(rng), dist(rng)};
        Vec3d proj = Project(a, b);  // Composante de a dans la direction de b
        Vec3d rej = Reject(a, b);    // Composante de a orthogonale à b
        ASSERT_TRUE(ApproxVec(proj + rej, a));  // La somme doit reconstituer a
    }
}

// TP6 - Semaine 2 : Vec4d projection perspective
TEST_CASE(Vec4dProjectionPerspective, Semaine2_TP3) {
    std::vector<Vec2d> proj;  // Stockage des projections 2D

    // Positionnement de la caméra à z = 2.0 pour une vue de face
    double z_cam = 2.0;
    for(auto& p : cube){
        p.z += z_cam;  // Translation du cube vers l'avant
        proj.push_back(ProjectPoint(p));  // Projection perspective simple
    }

    // Dessin des sommets du cube dans l'image (carrés rouges pour visibilité)
    for(const auto& p : proj) {
        int x = (int)p.x, y = (int)p.y;
        // Petit carré de 5x5 pixels autour de chaque sommet
        for(int dx = -2; dx <= 2; dx++)
            for(int dy = -2; dy <= 2; dy++)
                img.SetPixel(x+dx, y+dy, 255, 0, 0);  // Rouge
    }

    // Dessin des arêtes du cube
    for(auto edge : edges)
        img.DrawLine((int)proj[edge.x].x, (int)proj[edge.x].y,
                    (int)proj[edge.y].x, (int)proj[edge.y].y);

    // Sauvegarde de l'image au format PPM
    img.SavePPM("cube.ppm");
}

// TP7 - Semaine 3 : Mat4d inverse
TEST_CASE(Mat4dEtInverse, Semaine3_TP1) {
    Mat4d m, r, inv;

    // Tests sur 10 matrices aléatoires
    for(int t = 0; t < 10; t++) {
        // Génération d'une matrice 4x4 avec des valeurs aléatoires
        for(int i = 0; i < 4; i++)
            for(int j = 0; j < 4; j++)
                m(i, j) = dist(rng);

        // 1. Propriété : M × Identité = M (10 tests)
        r = m * Mat4d::Identity();
        ASSERT_TRUE(ApproxMat(r, m));  // La matrice doit rester inchangée

        // 2. Propriété : M × M⁻¹ = Identité (10 tests)
        // Seuls les matrices inversibles sont testées
        if(Inverse(m, inv))  // Calcule l'inverse si possible
            ASSERT_TRUE(ApproxMat(m * inv, Mat4d::Identity(), 1e-10f));  // Produit doit être l'identité
    }

    // 3. Test de matrice singulière : l'inverse doit échouer
    m = Mat4d::Identity();
    // Rendre la matrice singulière en dupliquant une ligne
    for(int j = 0; j < 4; j++)
        m(1, j) = m(0, j);  // Ligne 1 = ligne 0
    ASSERT_TRUE(!Inverse(m, inv));  // L'inverse ne doit pas exister

    // 4. Rotation autour de l'axe Y de π/2 appliquée au vecteur (1,0,0,1) doit donner (0,0,-1,1)
    r = Mat4d::RotateAxis({0,1,0}, NKENTSEU_PI_DOUBLE / 2.0f);  // Rotation de 90° autour de Y
    Vec4d s = {1,0,0,1}, q = r * s;  // Application de la rotation

    ASSERT_TRUE(approxEq(q.x, 0.0));   // x doit devenir 0
    ASSERT_TRUE(approxEq(q.y, 0.0));   // y reste 0
    ASSERT_TRUE(approxEq(q.z, -1.0));  // z devient -1
}

// TP8 - Semaine 3 : Rasteriseur rotation cube
TEST_CASE(RasteriseurRotationCube, Semaine3_TP2) {
    // Génération de 10 frames d'animation
    for(int frame = 0; frame < 10; frame++){
        img = NkImage(width, height);  // Nouvelle image pour chaque frame
        double angle = frame * 0.3;    // Angle de rotation progressif
        Mat4d R = Mat4d::RotateAxis(up, angle);  // Matrice de rotation autour de l'axe Y
        std::vector<Vec3d> screen;  // Positions projetées à l'écran

        // Application des transformations à chaque sommet du cube
        for(auto v : cube){
            Vec4d p = P * (V * (R * v));  // Rotation → Vue → Projection
            screen.push_back(ProjectToScreen(p, width, height));  // Projection à l'écran
        }

        // Dessin des arêtes du cube en blanc
        for(auto edge : edges)
            img.DrawLine((int)screen[edge.x].x, (int)screen[edge.x].y,
                        (int)screen[edge.y].x, (int)screen[edge.y].y, 255);

        // Sauvegarde de chaque frame
        img.SavePPM("frame_TP8_"+std::to_string(frame)+".ppm");
    }
}


// TP9 - Semaine 3 : TRS décomposition
TEST_CASE(ConstructionDecompositionTRS, Semaine3_TP3) {
    // Changement de distribution pour des valeurs plus variées
    dist = std::uniform_real_distribution<double>(-5.0, 5.0);

    // Tests sur 20 matrices TRS aléatoires
    for(int t = 0; t < 20; t++){
        // Génération de paramètres aléatoires
        Vec3d outT{dist(rng), dist(rng), dist(rng)};        // Translation
        Vec3d outR{dist(rng), dist(rng), dist(rng)};        // Rotation (angles)
        Vec3d outS{dist(rng) + 6, dist(rng) + 6, dist(rng) + 6};  // Échelle (positive)

        // 1. Construction de la matrice TRS
        Mat4d M = TRS(outT, outR, outS);

        // 2. Décomposition de la matrice pour récupérer les paramètres
        Vec3d T2, R2, S2;
        DecomposeTRS(M, T2, R2, S2);

        // 3. Vérification que les paramètres récupérés correspondent aux originaux
        ASSERT_TRUE(ApproxVec(outT, T2));        // Translation
        ASSERT_TRUE(ApproxVec(outS, S2));        // Échelle
        // Rotation : tolérance plus large due aux ambiguïtés dans les angles
        ASSERT_TRUE(ApproxVec(outR, R2, 5.0));  // Rotation avec tolérance
    }
}


// TP10 - Semaine 4 : Quaternions
TEST_CASE(OperationsQuaternions, Semaine4_TP1) {
    Mat3d m1, m2, m3;
    Quat q1, q2, q3;

    // 1. Vérification de la rotation et de FromAxisAngle avec π
    // Rotation de 90° autour de l'axe Y transforme (1,0,0) en (0,0,-1)
    Vec3d i = {1,0,0};
    q1 = FromAxisAngle({0,1,0}, NKENTSEU_PI_DOUBLE / 2.0f);  // Quaternion pour rotation Y 90°
    Vec3d j = Rotate(q1, i);  // Application de la rotation

    ASSERT_TRUE(std::fabs(j.x - 0.0) < kEps);  // x devient 0
    ASSERT_TRUE(std::fabs(j.y - 0.0) < kEps);  // y reste 0
    ASSERT_TRUE(std::fabs(j.z + 1.0) < kEps);  // z devient -1

    // 2. Aller-retour Quat ↔ Mat3d (50 tests)
    // Changement de distribution pour des quaternions unitaires
    dist = std::uniform_real_distribution<double>(-1.0, 1.0);

    for(int t = 0; t < 50; t++){
        // Génération d'un quaternion aléatoire
        q1 = { dist(rng), dist(rng), dist(rng), dist(rng) };
        q1 = q1.Normalized();  // Normalisation pour obtenir un quaternion unitaire

        // Conversion Quat → Mat3d → Quat
        m1 = ToMat3(q1);       // Matrice de rotation équivalente
        q2 = FromMat3(m1);     // Reconstruction du quaternion
        q2 = q2.Normalized();  // Renormalisation

        // Vérification de la correspondance
        ASSERT_TRUE(ApproxQuat(q1, q2, 1e-4f));
    }

    // 3. Vérification que Quat × Quat.Inverse() = Identité (50 tests)
    for(int t = 0; t < 50; t++){
        // Génération d'un quaternion aléatoire
        q1 = { dist(rng), dist(rng), dist(rng), dist(rng) };
        q1 = q1.Normalized();  // Quaternion unitaire
        q2 = q1.Inverse();     // Son inverse
        q3 = q1 * q2;          // Produit

        // Le produit doit être l'identité
        ASSERT_TRUE(ApproxQuat(q3, Quat::Identity(), 1e-4f));
    }
}


// TP11 - Semaine 4 : Animation SLERP
TEST_CASE(AnimationInterpolationSLERP, Semaine4_TP2) {
    Quat q1, q2, q3;

    // 1. Animation avec SLERP sur 60 frames
    // Interpolation sphérique entre deux orientations
    q1 = FromAxisAngle({0,1,0}, 0);                    // Orientation initiale (0°)
    q2 = FromAxisAngle({0,1,0}, NKENTSEU_PI_DOUBLE);   // Orientation finale (180°)

    for(int frame = 0; frame < 60; frame++){
        double t = frame / 59.0;  // Paramètre d'interpolation [0,1]
        Quat q = Slerp(q1, q2, t);  // Interpolation SLERP
        Mat4d R = FromRT(ToMat3(q), {0,0,0});  // Matrice de rotation

        img = NkImage(width, height);
        std::vector<Vec3d> screen;
        // Application au cube et projection
        for(auto v : cube){
            Vec4d p = P * (V * (R * v));
            screen.push_back(ProjectToScreen(p, width, height));
        }

        // Dessin du cube
        for(auto edge : edges)
            img.DrawLine((int)screen[edge.x].x, (int)screen[edge.x].y,
                        (int)screen[edge.y].x, (int)screen[edge.y].y, 255);
        img.SavePPM("Slerp_frame_TP11_"+std::to_string(frame)+".ppm");
    }

    // 2. Animation avec LERP sur 60 frames (pour comparaison)
    // Interpolation linéaire entre les mêmes orientations
    for(int frame = 0; frame < 60; frame++){
        double t = frame / 59.0;
        Quat q = Lerp(q1, q2, t);  // Interpolation LERP
        Mat4d R = FromRT(ToMat3(q), {0,0,0});

        img = NkImage(width, height);
        std::vector<Vec3d> screen;
        for(auto v : cube){
            Vec4d p = P * (V * (R * v));
            screen.push_back(ProjectToScreen(p, width, height));
        }

        for(auto edge : edges)
            img.DrawLine((int)screen[edge.x].x, (int)screen[edge.x].y,
                        (int)screen[edge.y].x, (int)screen[edge.y].y, 255);
        img.SavePPM("Lerp_frame_TP11_"+std::to_string(frame)+".ppm");
    }
}
