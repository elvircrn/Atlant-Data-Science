#include <bits/stdc++.h>

using namespace std;

int main()
{
    freopen("Data/ratings.csv", "r", stdin);
    freopen("Data/ratings1.csv", "w", stdout);

    char c;
    while ((c = getchar()) != '\n')
        putchar(c);
    putchar('\n');

    while ((c = getchar()) != EOF)
    {
        putchar(c);
        int comma{};
        while ((c = getchar()) != EOF)
        {
            putchar(c);
            comma += (c == ',');
            if (comma == 3)
                break;
        }
    }

    return 0;
}
