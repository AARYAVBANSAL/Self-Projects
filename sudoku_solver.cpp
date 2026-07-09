// Sudoku Solver Algorithm in C++
// Self Project (Mar '24 - Apr '24)
//
// - Recursive backtracking algorithm to solve 9x9 Sudoku puzzles while
//   ensuring valid number placement (row / column / 3x3 box constraints).
// - Handles unsolvable cases by backtracking when no valid digit can be
//   placed, and reports "No solution exists" instead of crashing/looping.
//
// Build:   g++ -std=c++17 -O2 -o sudoku_solver sudoku_solver.cpp
// Run:     ./sudoku_solver               (uses the built-in sample puzzle)
//          ./sudoku_solver puzzle.txt    (reads a puzzle from a file)
//
// Puzzle file format: 9 lines of 9 characters (digits 1-9, and '0' or '.'
// for empty cells). Whitespace is ignored.

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>

using Grid = std::vector<std::vector<int>>;
constexpr int SIZE = 9;
constexpr int BOX = 3;

class SudokuSolver {
public:
    explicit SudokuSolver(Grid grid) : board(std::move(grid)) {}

    // Attempts to solve the puzzle in place. Returns true if a solution
    // was found, false if the puzzle is unsolvable.
    bool solve() {
        nodesExplored = 0;
        return backtrack();
    }

    const Grid& getBoard() const { return board; }
    long long getNodesExplored() const { return nodesExplored; }

    void print() const {
        for (int r = 0; r < SIZE; ++r) {
            if (r % BOX == 0 && r != 0)
                std::cout << "------+-------+------\n";
            for (int c = 0; c < SIZE; ++c) {
                if (c % BOX == 0 && c != 0) std::cout << "| ";
                std::cout << (board[r][c] == 0 ? std::string(".") : std::to_string(board[r][c])) << ' ';
            }
            std::cout << '\n';
        }
    }

    // Validates that a (possibly partially filled) starting grid does not
    // already violate Sudoku rules.
    bool isInputValid() const {
        for (int r = 0; r < SIZE; ++r) {
            for (int c = 0; c < SIZE; ++c) {
                int val = board[r][c];
                if (val == 0) continue;
                // Temporarily clear the cell to check placement validity.
                Grid copy = board;
                copy[r][c] = 0;
                if (!isSafeOn(copy, r, c, val)) return false;
            }
        }
        return true;
    }

private:
    Grid board;
    long long nodesExplored = 0;

    // Finds the next empty cell (0). Returns false if the board is full.
    bool findEmptyCell(int &row, int &col) const {
        for (row = 0; row < SIZE; ++row)
            for (col = 0; col < SIZE; ++col)
                if (board[row][col] == 0) return true;
        return false;
    }

    bool isSafeOn(const Grid &g, int row, int col, int val) const {
        // Row and column check.
        for (int i = 0; i < SIZE; ++i) {
            if (g[row][i] == val) return false;
            if (g[i][col] == val) return false;
        }
        // 3x3 box check.
        int boxRow = (row / BOX) * BOX;
        int boxCol = (col / BOX) * BOX;
        for (int r = boxRow; r < boxRow + BOX; ++r)
            for (int c = boxCol; c < boxCol + BOX; ++c)
                if (g[r][c] == val) return false;
        return true;
    }

    bool isSafe(int row, int col, int val) const {
        return isSafeOn(board, row, col, val);
    }

    // Core recursive backtracking routine.
    bool backtrack() {
        int row, col;
        if (!findEmptyCell(row, col)) return true; // Solved: no empty cells left.

        for (int val = 1; val <= 9; ++val) {
            ++nodesExplored;
            if (isSafe(row, col, val)) {
                board[row][col] = val;      // Place a candidate value.
                if (backtrack()) return true;
                board[row][col] = 0;        // Undo (backtrack) on failure.
            }
        }
        return false; // No valid digit works here -> trigger backtracking upstream.
    }
};

// Reads a 9x9 grid from a text file. '.', '0', or blank all mean empty.
Grid readGridFromFile(const std::string &path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Could not open file: " + path);

    Grid grid(SIZE, std::vector<int>(SIZE, 0));
    std::string line;
    int row = 0;
    while (row < SIZE && std::getline(in, line)) {
        int col = 0;
        for (char ch : line) {
            if (col >= SIZE) break;
            if (ch == '.' || ch == '0') { grid[row][col++] = 0; }
            else if (ch >= '1' && ch <= '9') { grid[row][col++] = ch - '0'; }
            // any other character (spaces, commas) is skipped
        }
        if (col > 0) ++row;
    }
    if (row != SIZE) throw std::runtime_error("Puzzle file did not contain 9 valid rows.");
    return grid;
}

Grid sampleGrid() {
    return {
        {5,3,0, 0,7,0, 0,0,0},
        {6,0,0, 1,9,5, 0,0,0},
        {0,9,8, 0,0,0, 0,6,0},

        {8,0,0, 0,6,0, 0,0,3},
        {4,0,0, 8,0,3, 0,0,1},
        {7,0,0, 0,2,0, 0,0,6},

        {0,6,0, 0,0,0, 2,8,0},
        {0,0,0, 4,1,9, 0,0,5},
        {0,0,0, 0,8,0, 0,7,9}
    };
}

int main(int argc, char* argv[]) {
    try {
        Grid grid = (argc > 1) ? readGridFromFile(argv[1]) : sampleGrid();

        SudokuSolver solver(grid);

        std::cout << "Input puzzle:\n";
        solver.print();
        std::cout << '\n';

        if (!solver.isInputValid()) {
            std::cout << "The given puzzle already violates Sudoku rules. No solution exists.\n";
            return 1;
        }

        auto start = std::chrono::high_resolution_clock::now();
        bool solved = solver.solve();
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();

        if (solved) {
            std::cout << "Solved puzzle:\n";
            solver.print();
            std::cout << "\nNodes explored: " << solver.getNodesExplored()
                      << " | Time: " << ms << " ms\n";
        } else {
            std::cout << "No solution exists for the given puzzle.\n";
        }
        return solved ? 0 : 1;

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 2;
    }
}
