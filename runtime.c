#include <stdio.h>

extern void PL0MAIN();

int _pl0_read()
{
    int n = 0;
    scanf("%d", &n);
    return n;
}

void _pl0_writeln(int x)
{
    printf("%d\n", x);
}

int main()
{
    PL0MAIN();
    return 0;
}
