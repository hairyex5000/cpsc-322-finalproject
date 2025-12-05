import copy
import csv
from tabulate import tabulate


class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names (list of str): M column names
        data (list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Parameters:
            column_names (list of str): initial M column names (None if empty)
            data (list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure."""
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            tuple: (N, M) where N is number of rows and M is number of columns
        """
        return len(self.data), len(self.column_names)

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Parameters:
            col_identifier (str or int): string for a column name or int
                for a column index
            include_missing_values (bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Raises:
            ValueError: if col_identifier is invalid
        """
        out = []

        # Get the column as an index regardless of the arg format
        col_index = -1

        if type(col_identifier) is int:
            # Validate
            if col_identifier < 0 or col_identifier >= len(self.column_names):
                raise ValueError()

            col_index = col_identifier
        else:
            # Will raise ValueError for us
            col_index = self.column_names.index(col_identifier)

        for row in self.data:
            # Skip NA if needed
            if row[col_index] == "NA" and include_missing_values == False:
                continue

            out.append(row[col_index])

        return out

    def replace_column(self, col_identifier, new_col):
        """Replaces a column in the table data with new values.

        Parameters:
            col_identifier (str or int): string for a column name or int
                for a column index
            new_col (list of obj): new column values to replace the existing column with

        Raises:
            ValueError: if col_identifier is invalid
        """

        # Get the column as an index regardless of the arg format
        col_index = -1

        if type(col_identifier) is int:
            # Validate
            if col_identifier < 0 or col_identifier >= len(self.column_names):
                raise ValueError()

            col_index = col_identifier
        else:
            # Will raise ValueError for us
            col_index = self.column_names.index(col_identifier)

        for i in range(len(self.data)):
            self.data[i][col_index] = new_col[i]

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leaves values as-is that cannot be converted to numeric.
        """
        for i in range(len(self.data)):
            for j in range(len(self.column_names)):
                try:
                    self.data[i][j] = float(self.data[i][j])
                except:
                    pass

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Parameters:
            row_indexes_to_drop (list of int): list of row indexes to remove from the table data.
        """
        # Iterate back to front to not affect indices of elements we will delete next
        for index in range(len(self.data) - 1, -1, -1):
            if index in row_indexes_to_drop:
                del self.data[index]

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Parameters:
            filename (str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: returns self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Uses the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load.
        """

        with open(filename, "r") as in_file:
            reader = csv.reader(in_file)

            self.column_names = next(reader)

            for row in reader:
                self.data.append(row)

        self.convert_to_numeric()

        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Parameters:
            filename (str): relative path for the CSV file to save the contents to.

        Notes:
            Uses the csv module.
        """
        with open(filename, "w") as out_file:
            writer = csv.writer(out_file)

            writer.writerow(self.column_names)
            writer.writerows(self.data)

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Parameters:
            key_column_names (list of str): column names to use as row keys.

        Returns:
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
            The first instance of a row is not considered a duplicate.
        """
        # Use a set since we only want one instance of an index in this
        out = set()

        # Transform names to indices
        key_indices = [self.column_names.index(
            name) for name in key_column_names]

        # For every row, check all rows after it and see if all keys match. If so, the later row is a duplicate
        for i in range(len(self.data)):
            for j in range(i + 1, len(self.data)):
                duplicate = all(
                    self.data[i][key] == self.data[j][key] for key in key_indices)

                if duplicate:
                    out.add(j)

        # Sort due to weirdness in the ordering of the output
        return sorted(list(out))

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA")."""
        # Iterate back to front to not affect indices of elements we will delete next
        for index in range(len(self.data) - 1, -1, -1):
            for col in self.data[index]:
                if col == "NA":
                    del self.data[index]
                    break

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
        by the column's original average.

        Parameters:
            col_name (str): name of column to fill with the original average (of the column).
        """
        col_index = self.column_names.index(col_name)
        col = self.get_column(col_index, False)

        avg = sum(col) / len(col)

        for row in self.data:
            if row[col_index] == "NA":
                row[col_index] = avg

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
            min: minimum of the column
            max: maximum of the column
            mid: mid-value (AKA mid-range) of the column
            avg: mean of the column
            median: median of the column

        Parameters:
            col_names (list of str): names of the numeric columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]

        Notes:
            Missing values in the columns to compute summary stats
            should be ignored.
            Assumes col_names only contains the names of columns with numeric data.
        """
        data = []

        for col_name in col_names:
            col = self.get_column(col_name, False)

            # Do nothing for an empty list
            if len(col) == 0:
                continue

            # Makes min, max, median easier
            col.sort()

            min_v = col[0]
            max_v = col[len(col) - 1]
            mid = (min_v + max_v) / 2
            avg = sum(col) / len(col)

            median = 0
            if len(col) % 2 == 1:
                median = col[len(col) // 2]
            else:
                # If there is an even number of rows, average the two middle values
                i = len(col) // 2 - 1
                median = (col[i] + col[i + 1]) / 2

            data.append([col_name, min_v, max_v, mid, avg, median])

        return MyPyTable(["attribute", "min", "max", "mid", "avg", "median"], data)

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
        with other_table based on key_column_names.

        Parameters:
            other_table (MyPyTable): the second table to join this table with.
            key_column_names (list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        out = []

        # Create the resulting headers first
        header = [*self.column_names]

        # Add headers from other (w/o keys)
        for col in other_table.column_names:
            if col in key_column_names:
                continue

            header.append(col)

        # Indices may be in different locations for each table
        key1_indices = [self.column_names.index(
            name) for name in key_column_names]
        key2_indices = [other_table.column_names.index(
            name) for name in key_column_names]

        for row1 in self.data:
            for row2 in other_table.data:
                # Are all keys identical in both rows?
                same = all([row1[key1_indices[i]] == row2[key2_indices[i]]
                           for i in range(len(key_column_names))])

                if same:
                    # Add data from self
                    row3 = [*row1]

                    # Add data from other
                    for i in range(len(row2)):
                        if other_table.column_names[i] in key_column_names:
                            continue

                        row3.append(row2[i])

                    out.append(row3)

        return MyPyTable(header, out)

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
        other_table based on key_column_names.

        Parameters:
            other_table (MyPyTable): the second table to join this table with.
            key_column_names (list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pads attributes with missing values with "NA".
        """
        out = []

        # Create the resulting headers first
        header = [*self.column_names]

        # Add headers from other (w/o keys)
        for col in other_table.column_names:
            if col in key_column_names:
                continue

            header.append(col)

        # Indices may be in different locations for each table
        key1_indices = [self.column_names.index(
            name) for name in key_column_names]
        key2_indices = [other_table.column_names.index(
            name) for name in key_column_names]

        for row1 in self.data:
            matched = False

            for row2 in other_table.data:
                # Are all keys identical in both rows?
                same = all([row1[key1_indices[i]] == row2[key2_indices[i]]
                           for i in range(len(key_column_names))])

                row3 = []

                if same:
                    matched = True

                    # Add data from self
                    row3 = [*row1]

                    # Add data from other
                    for i in range(len(row2)):
                        if other_table.column_names[i] in key_column_names:
                            continue

                        row3.append(row2[i])

                    out.append(row3)

            # If there were no matches, add the row in with NA's where needed
            if matched:
                continue

            row3 = []

            for col in header:
                if col in self.column_names:
                    i = self.column_names.index(col)
                    row3.append(row1[i])
                else:
                    row3.append("NA")

            out.append(row3)

        # Now add the rows from the other table that aren't in the out table
        for row2 in other_table.data:
            matched = False

            for row3 in out:
                # Are all keys identical in both rows?
                same = all([row3[key1_indices[i]] == row2[key2_indices[i]]
                           for i in range(len(key_column_names))])

                if same:
                    matched = True
                    break

            if matched:
                continue

            # Add row to out, add NAs where needed
            row3 = []

            for col in header:
                if col in other_table.column_names:
                    i = other_table.column_names.index(col)
                    row3.append(row2[i])
                else:
                    row3.append("NA")

            out.append(row3)

        return MyPyTable(header, out)
