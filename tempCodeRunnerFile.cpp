#include <iostream>
#include <cmath>
using namespace std;

int main() {
    int n;
    cout << "Enter the limit: ";
    cin >> n;

    for (int i = 1; i <= n; i++) {
        int temp = i;
        int result = 0;
        int digits = 0;

        // Count digits
        int t = i;
        while (t != 0) {
            t /= 10;
            digits++;
        }

        temp = i;

        // Armstrong check
        while (temp != 0) {
            int remainder = temp % 10;
            result += (int)pow(remainder, digits); // important fix
            temp /= 10;
        }

        if (result == i) {
            cout << i << " ";
        }
    }

    return 0;
}