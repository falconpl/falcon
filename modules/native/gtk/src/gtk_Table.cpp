/**
 *  \file gtk_Table.cpp
 */

#include "gtk_Table.hpp"

#include <gtk/gtk.h>

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void Table::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Table = mod->addClass( "GtkTable", &Table::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkContainer" ) );
    c_Table->getClassDef()->addInheritance( in );

    c_Table->getClassDef()->factory( &Table::factory );

    Gtk::MethodTab methods[] =
    {
    { "resize",                 &Table::resize },
    { "attach",                 &Table::attach },
    { "attach_defaults",        &Table::attach_defaults },
    { "set_row_spacing",        &Table::set_row_spacing },
    { "set_col_spacing",        &Table::set_col_spacing },
    { "set_row_spacings",       &Table::set_row_spacings },
    { "set_col_spacings",       &Table::set_col_spacings },
    { "set_homogeneous",        &Table::set_homogeneous },
    { "get_default_row_spacing",&Table::get_default_row_spacing },
    { "get_homogeneous",        &Table::get_homogeneous },
    { "get_row_spacing",        &Table::get_row_spacing },
    { "get_col_spacing",        &Table::get_col_spacing },
    { "get_default_col_spacing",&Table::get_default_col_spacing },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Table, meth->name, meth->cb );
}


Table::Table( const Falcon::CoreClass* gen, const GtkTable* tab )
    :
    Gtk::CoreGObject( gen, (GObject*) tab )
{}


Falcon::CoreObject* Table::factory( const Falcon::CoreClass* gen, void* tab, bool )
{
    return new Table( gen, (GtkTable*) tab );
}


/*#
    @class GtkTable
    @brief Pack widgets in regular patterns
    @optparam rows The number of rows the new table should have (default 0).
    @optparam columns The number of columns the new table should have (default 0).
    @optparam homogeneous (default false) If set to true, all table cells are resized to the size of the cell containing the largest widget.

    The GtkTable functions allow the programmer to arrange widgets in rows and columns,
    making it easy to align many widgets next to each other, horizontally and vertically.

    Widgets can be added to a table using attach() or the more convenient
    (but slightly less flexible) attach_defaults().

    To alter the space next to a specific row, use set_row_spacing(), and
    for a column, set_col_spacing().

    The gaps between all rows or columns can be changed by calling set_row_spacings()
    or set_col_spacings() respectively.

    set_homogeneous(), can be used to set whether all cells in the table will resize
    themselves to the size of the largest widget in the table.
 */
FALCON_FUNC Table::init( VMARG )
{
    MYSELF;

    if ( self->getObject() )
        return;

    Item* i_rows = vm->param( 0 );
    Item* i_cols = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( i_rows )
    {
        if ( i_rows->isNil() || !i_rows->isInteger() )
            throw_inv_params( "[I,I,B]" );
    }
    if ( i_cols )
    {
        if ( i_cols->isNil() || !i_cols->isInteger() )
            throw_inv_params( "[I,I,B]" );
    }
#endif
    int rows = i_rows ? i_rows->asInteger() : 0;
    int cols = i_cols ? i_cols->asInteger() : 0;
    GtkWidget* wdt;

    Item* i_homog = vm->param( 2 );
    if ( i_homog )
    {
#ifndef NO_PARAMETER_CHECK
        if ( i_homog->isNil() || !i_homog->isBoolean() )
            throw_inv_params( "I,I[,B]" );
#endif
        wdt = gtk_table_new( rows, cols, i_homog->asBoolean() ? TRUE : FALSE );
    }
    else
        wdt = gtk_table_new( rows, cols, FALSE );

    self->setObject( (GObject*) wdt );
}


/*#
    @method resize GtkTable
    @brief Resizes the table.
    @param rows The new number of rows.
    @params columns The new number of columns.

    If you need to change a table's size after it has been created, this function
    allows you to do so.
 */
FALCON_FUNC Table::resize( VMARG )
{
    Item* i_rows = vm->param( 0 );
    Item* i_cols = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_rows || i_rows->isNil() || !i_rows->isInteger()
        || !i_cols || i_cols->isNil() || !i_cols->isInteger() )
        throw_inv_params( "I,I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_table_resize( (GtkTable*)_obj, i_rows->asInteger(), i_cols->asInteger() );
}


/*#
    @method attach GtkTable
    @brief Adds a widget to a table.
    @param child The widget to add.
    @param left_attach the column number to attach the left side of a child widget to.
    @param right_attach the column number to attach the right side of a child widget to.
    @param top_attach the row number to attach the top of a child widget to.
    @param bottom_attach the row number to attach the bottom of a child widget to.
    @param xoptions (GtkAttachOptions) Used to specify the properties of the child widget when the table is resized.
    @param yoptions (GtkAttachOptions) The same as xoptions, except this field determines behaviour of vertical resizing.
    @param xpadding An integer value specifying the padding on the left and right of the widget being added to the table.
    @param ypadding The amount of padding above and below the child widget.

    The number of 'cells' that a widget will occupy is specified by left_attach,
    right_attach, top_attach and bottom_attach. These each represent the leftmost,
    rightmost, uppermost and lowest column and row numbers of the table.
    (Columns and rows are indexed from zero).
 */
FALCON_FUNC Table::attach( VMARG )
{
    Item* i_wdt = vm->param( 0 );
    Item* i_left = vm->param( 1 );
    Item* i_right = vm->param( 2 );
    Item* i_top = vm->param( 3 );
    Item* i_bottom = vm->param( 4 );
    Item* i_xopt = vm->param( 5 );
    Item* i_yopt = vm->param( 6 );
    Item* i_xpad = vm->param( 7 );
    Item* i_ypad = vm->param( 8 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_wdt || i_wdt->isNil()
        || !IS_DERIVED( i_wdt, GtkWidget )
        || !i_left || i_left->isNil() || !i_left->isInteger()
        || !i_right || i_right->isNil() || !i_right->isInteger()
        || !i_top || i_top->isNil() || !i_top->isInteger()
        || !i_bottom || i_bottom->isNil() || !i_bottom->isInteger()
        || !i_xopt || i_xopt->isNil() || !i_xopt->isInteger()
        || !i_yopt || i_yopt->isNil() || !i_yopt->isInteger()
        || !i_xpad || i_xpad->isNil() || !i_xpad->isInteger()
        || !i_ypad || i_ypad->isNil() || !i_ypad->isInteger() )
        throw_inv_params( "GtkWidget,I,I,I,I,I,I,I" );
#endif
    GtkWidget* wdt = (GtkWidget*) COREGOBJECT( i_wdt )->getObject();
    MYSELF;
    GET_OBJ( self );
    gtk_table_attach( (GtkTable*)_obj, wdt, i_left->asInteger(), i_right->asInteger(),
        i_top->asInteger(), i_bottom->asInteger(),
        (GtkAttachOptions) i_xopt->asInteger(), (GtkAttachOptions) i_yopt->asInteger(),
        i_xpad->asInteger(), i_ypad->asInteger() );
}


/*#
    @method attach_defaults GtkTable
    @brief Adds a widget to a table.
    @param child The widget to add.
    @param left_attach the column number to attach the left side of a child widget to.
    @param right_attach the column number to attach the right side of a child widget to.
    @param top_attach the row number to attach the top of a child widget to.
    @param bottom_attach the row number to attach the bottom of a child widget to.

    As there are many options associated with attach(), this convenience function
    provides the programmer with a means to add children to a table with identical
    padding and expansion options. The values used for the GtkAttachOptions are
    GTK_EXPAND | GTK_FILL, and the padding is set to 0.
 */
FALCON_FUNC Table::attach_defaults( VMARG )
{
    Item* i_wdt = vm->param( 0 );
    Item* i_left = vm->param( 1 );
    Item* i_right = vm->param( 2 );
    Item* i_top = vm->param( 3 );
    Item* i_bottom = vm->param( 4 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_wdt || i_wdt->isNil()
        || !IS_DERIVED( i_wdt, GtkWidget )
        || !i_left || i_left->isNil() || !i_left->isInteger()
        || !i_right || i_right->isNil() || !i_right->isInteger()
        || !i_top || i_top->isNil() || !i_top->isInteger()
        || !i_bottom || i_bottom->isNil() || !i_bottom->isInteger() )
        throw_inv_params( "GtkWidget,I,I,I,I" );
#endif
    GtkWidget* wdt = (GtkWidget*) COREGOBJECT( i_wdt )->getObject();
    MYSELF;
    GET_OBJ( self );
    gtk_table_attach_defaults( (GtkTable*)_obj, wdt, i_left->asInteger(), i_right->asInteger(),
        i_top->asInteger(), i_bottom->asInteger() );
}


/*#
    @method set_row_spacing GtkTable
    @brief Changes the space between a given table row and the subsequent row.
    @param row row number whose spacing will be changed.
    @param spacing number of pixels that the spacing should take up.
 */
FALCON_FUNC Table::set_row_spacing( VMARG )
{
    Item* i_row = vm->param( 0 );
    Item* i_spac = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_row || i_row->isNil() || !i_row->isInteger()
        || !i_spac || i_spac->isNil() || !i_spac->isInteger() )
        throw_inv_params( "I,I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_table_set_row_spacing( (GtkTable*)_obj, i_row->asInteger(), i_spac->asInteger() );
}


/*#
    @method set_col_spacing GtkTable
    @brief Alters the amount of space between a given table column and the following column.
    @param column the column whose spacing should be changed.
    @param spacing number of pixels that the spacing should take up.
 */
FALCON_FUNC Table::set_col_spacing( VMARG )
{
    Item* i_col = vm->param( 0 );
    Item* i_spac = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_col || i_col->isNil() || !i_col->isInteger()
        || !i_spac || i_spac->isNil() || !i_spac->isInteger() )
        throw_inv_params( "I,I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_table_set_col_spacing( (GtkTable*)_obj, i_col->asInteger(), i_spac->asInteger() );
}


/*#
    @method set_row_spacings GtkTable
    @brief Sets the space between every row in table equal to spacing.
    @param spacing the number of pixels of space to place between every row in the table.
 */
FALCON_FUNC Table::set_row_spacings( VMARG )
{
    Item* i_spac = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_spac || i_spac->isNil() || !i_spac->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_table_set_row_spacings( (GtkTable*)_obj, i_spac->asInteger() );
}


/*#
    @method set_col_spacings GtkTable
    @brief Sets the space between every column in table equal to spacing.
    @param spacing the number of pixels of space to place between every column in the table.
 */
FALCON_FUNC Table::set_col_spacings( VMARG )
{
    Item* i_spac = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_spac || i_spac->isNil() || !i_spac->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_table_set_col_spacings( (GtkTable*)_obj, i_spac->asInteger() );
}


/*#
    @method set_homogeneous GtkTable
    @brief Changes the homogenous property of table cells, ie. whether all cells are an equal size or not.
    @param homogeneous Set to true to ensure all table cells are the same size. Set to false if this is not your desired behaviour.
 */
FALCON_FUNC Table::set_homogeneous( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_table_set_homogeneous( (GtkTable*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method get_default_row_spacing GtkTable
    @brief Gets the default row spacing for the table. This is the spacing that will be used for newly added rows.
    @return the default row spacing
 */
FALCON_FUNC Table::get_default_row_spacing( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_table_get_default_row_spacing( (GtkTable*)_obj ) );
}


/*#
    @method get_homogeneous GtkTable
    @brief Returns whether the table cells are all constrained to the same width and height.
    @return (boolean) true if the cells are all constrained to the same size
 */
FALCON_FUNC Table::get_homogeneous( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_table_get_homogeneous( (GtkTable*)_obj ) );
}


/*#
    @method get_row_spacing GtkTable
    @brief Gets the amount of space between row row, and row row + 1.
    @param row a row in the table, 0 indicates the first row
    @return the row spacing
 */
FALCON_FUNC Table::get_row_spacing( VMARG )
{
    Item* i_row = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_row || i_row->isNil() || !i_row->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_table_get_row_spacing( (GtkTable*)_obj, i_row->asInteger() ) );
}


/*#
    @method get_col_spacing GtkTable
    @brief Gets the amount of space between column col, and column col + 1.
    @param column a column in the table, 0 indicates the first column
    @return the column spacing
 */
FALCON_FUNC Table::get_col_spacing( VMARG )
{
    Item* i_col = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_col || i_col->isNil() || !i_col->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_table_get_col_spacing( (GtkTable*)_obj, i_col->asInteger() ) );
}


/*#
    @method get_default_col_spacing GtkTable
    @brief Gets the default column spacing for the table.
    @return the default column spacing

    This is the spacing that will be used for newly added columns.
 */
FALCON_FUNC Table::get_default_col_spacing( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_table_get_default_col_spacing( (GtkTable*)_obj ) );
}


} // Gtk
} // Falcon
