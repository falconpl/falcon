/**
 *  \file gtk_ComboBox.cpp
 */

#include "gtk_ComboBox.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void ComboBox::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_ComboBox = mod->addClass( "GtkComboBox", &ComboBox::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkBin" ) );
    c_ComboBox->getClassDef()->addInheritance( in );

    c_ComboBox->setWKS( true );
    c_ComboBox->getClassDef()->factory( &ComboBox::factory );

    Gtk::MethodTab methods[] =
    {
    { "signal_changed",         &ComboBox::signal_changed },
    { "signal_move_active",     &ComboBox::signal_move_active },
    { "signal_podown",          &ComboBox::signal_popdown },
    { "signal_popup",           &ComboBox::signal_popup },
    //{ "new_with_model",         &ComboBox::new_with_model },
    { "get_wrap_width",         &ComboBox::get_wrap_width },
    { "set_wrap_width",         &ComboBox::set_wrap_width },
    { "get_row_span_column",    &ComboBox::get_row_span_column },
    { "set_row_span_column",    &ComboBox::set_row_span_column },
    { "get_column_span_column", &ComboBox::get_column_span_column },
    { "set_column_span_column", &ComboBox::set_column_span_column },
    { "get_active",             &ComboBox::get_active },
    { "set_active",             &ComboBox::set_active },
    //{ "get_active_iter",        &ComboBox::get_active_iter },
    //{ "set_active_iter",        &ComboBox::set_active_iter },
    //{ "get_model",              &ComboBox::get_model },
    //{ "set_model",              &ComboBox::set_model },
    { "new_text",               &ComboBox::new_text },
    { "append_text",            &ComboBox::append_text },
    { "insert_text",            &ComboBox::insert_text },
    { "prepend_text",           &ComboBox::prepend_text },
    { "remove_text",            &ComboBox::remove_text },
    { "get_active_text",        &ComboBox::get_active_text },
    { "popup",                  &ComboBox::popup },
    { "popdown",                &ComboBox::popdown },
    //{ "get_popup_accessible",   &ComboBox::get_popup_accessible },
    //{ "get_row_separator_func", &ComboBox::get_row_separator_func },
    //{ "set_row_separator_func", &ComboBox::set_row_separator_func },
    { "set_add_tearoffs",       &ComboBox::set_add_tearoffs },
    { "get_add_tearoffs",       &ComboBox::get_add_tearoffs },
    { "set_title",              &ComboBox::set_title },
    { "get_title",              &ComboBox::get_title },
    { "set_focus_on_click",     &ComboBox::set_focus_on_click },
    { "get_focus_on_click",     &ComboBox::get_focus_on_click },
    { "set_button_sensitivity", &ComboBox::set_button_sensitivity },
    { "get_button_sensitivity", &ComboBox::get_button_sensitivity },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_ComboBox, meth->name, meth->cb );
}


ComboBox::ComboBox( const Falcon::CoreClass* gen, const GtkComboBox* box )
    :
    Gtk::CoreGObject( gen )
{
    if ( box )
        setUserData( new GData( (GObject*) box ) );
}


Falcon::CoreObject* ComboBox::factory( const Falcon::CoreClass* gen, void* box, bool )
{
    return new ComboBox( gen, (GtkComboBox*) box );
}


/*#
    @class GtkComboBox
    @brief A widget used to choose from a list of items

    A GtkComboBox is a widget that allows the user to choose from a list of valid choices.
    The GtkComboBox displays the selected choice. When activated, the GtkComboBox displays
    a popup which allows the user to make a new choice. The style in which the selected
    value is displayed, and the style of the popup is determined by the current theme.
    It may be similar to a GtkOptionMenu, or similar to a Windows-style combo box.

    Unlike its predecessors GtkCombo and GtkOptionMenu, the GtkComboBox uses the
    model-view pattern; the list of valid choices is specified in the form of a
    tree model, and the display of the choices can be adapted to the data in the
    model by using cell renderers, as you would in a tree view. This is possible
    since GtkComboBox implements the GtkCellLayout interface. The tree model holding
    the valid choices is not restricted to a flat list, it can be a real tree, and
    the popup will reflect the tree structure.

    In addition to the model-view API, GtkComboBox offers a simple API which is
    suitable for text-only combo boxes, and hides the complexity of managing the
    data in a model. It consists of the functions new_text(),
    append_text(), insert_text(), prepend_text(),
    remove_text() and get_active_text().
 */
FALCON_FUNC ComboBox::init( VMARG )
{
    MYSELF;

    if ( self->getUserData() )
        return;

#ifndef NO_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    GtkWidget* wdt = gtk_combo_box_new();
    Gtk::internal_add_slot( (GObject*) wdt );
    self->setUserData( new GData( (GObject*) wdt ) );
}


/*#
    @method signal_changed GtkComboBox
    @brief Connect a VMSlot to the combo box changed signal and return it

    The changed signal is emitted when the active item is changed. The can be due
    to the user selecting a different item from the list, or due to a call to
    gtk_combo_box_set_active_iter(). It will also be emitted while typing into a
    GtkComboBoxEntry, as well as when selecting an item from the GtkComboBoxEntry's list.
 */
FALCON_FUNC ComboBox::signal_changed( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    Gtk::internal_get_slot( "changed", (void*) &ComboBox::on_changed, vm );
}


void ComboBox::on_changed( GtkComboBox* wdt, gpointer _vm )
{
    Gtk::internal_trigger_slot( (GObject*) wdt, "changed",
        "on_changed", (VMachine*)_vm );
}


/*#
    @method signal_move_active GtkComboBox
    @brief Connect a VMSlot to the combo box move-active signal and return it

    The move-active signal is a keybinding signal which gets emitted to move
    the active selection.

    The callback function is passed a GtkScrollType as argument.
 */
FALCON_FUNC ComboBox::signal_move_active( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    Gtk::internal_get_slot( "move_active", (void*) &ComboBox::on_move_active, vm );
}


void ComboBox::on_move_active( GtkComboBox* obj, GtkScrollType scrolltype, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "move_active", false );

    if ( !cs || cs->empty() )
        return;

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_move_active", it ) )
            {
                printf(
                "[GtkComboBox::on_move_active] invalid callback (expected callable)\n" );
                return;
            }
        }
        vm->pushParam( (int64) scrolltype );
        vm->callItem( it, 1 );
    }
    while ( iter.hasCurrent() );
}


/*#
    @method signal_popdown GtkComboBox
    @brief Connect a VMSlot to the combo box popdown signal and return it

    The popdown signal is a keybinding signal which gets emitted to popdown
    the combo box list.

    The default bindings for this signal are Alt+Up and Escape.
 */
FALCON_FUNC ComboBox::signal_popdown( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    Gtk::internal_get_slot( "popdown", (void*) &ComboBox::on_popdown, vm );
}


void ComboBox::on_popdown( GtkComboBox* wdt, gpointer _vm )
{
    Gtk::internal_trigger_slot( (GObject*) wdt, "popdown",
        "on_popdown", (VMachine*)_vm );
}


/*#
    @method signal_popup GtkComboBox
    @brief Connect a VMSlot to the combo box popup signal and return it

    The popup signal is a keybinding signal which gets emitted to popup the
    combo box list.

    The default binding for this signal is Alt+Down.
 */
FALCON_FUNC ComboBox::signal_popup( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    Gtk::internal_get_slot( "popup", (void*) &ComboBox::on_popup, vm );
}


void ComboBox::on_popup( GtkComboBox* wdt, gpointer _vm )
{
    Gtk::internal_trigger_slot( (GObject*) wdt, "popup",
        "on_popup", (VMachine*)_vm );
}


//FALCON_FUNC ComboBox::new_with_model( VMARG );


/*#
    @method get_wrap_width GtkComBox
    @brief Returns the wrap width which is used to determine the number of columns for the popup menu.
    @return the wrap width

    If the wrap width is larger than 1, the combo box is in table mode.
 */
FALCON_FUNC ComboBox::get_wrap_width( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_combo_box_get_wrap_width( (GtkComboBox*)_obj ) );
}


/*#
    @method set_wrap_width GtkComboBox
    @brief Sets the wrap width of combo_box to be width.
    @param width Preferred number of columns
    The wrap width is basically the preferred number of columns when you want the
    popup to be layed out in a table.
 */
FALCON_FUNC ComboBox::set_wrap_width( VMARG )
{
    Item* i_width = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_width || i_width->isNil() || !i_width->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_combo_box_set_wrap_width( (GtkComboBox*)_obj, i_width->asInteger() );
}


/*#
    @method get_row_span_column GtkComboBox
    @brief Returns the column with row span information for combo_box.
    @return the row span column.
 */
FALCON_FUNC ComboBox::get_row_span_column( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_combo_box_get_row_span_column( (GtkComboBox*)_obj ) );
}


/*#
    @method set_row_span_column GtkComboBox
    @brief Sets the column with row span information for combo_box to be row_span.
    @param row_span A column in the model passed during construction.

    The row span column contains integers which indicate how many rows an item should span.
 */
FALCON_FUNC ComboBox::set_row_span_column( VMARG )
{
    Item* i_span = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_span || i_span->isNil() || !i_span->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_combo_box_set_row_span_column( (GtkComboBox*)_obj, i_span->asInteger() );
}


/*#
    @method get_column_span_column GtkComboBox
    @brief Returns the column with column span information for combo_box.
    @return the column span column
 */
FALCON_FUNC ComboBox::get_column_span_column( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_combo_box_get_column_span_column( (GtkComboBox*)_obj ) );
}


/*#
    @method set_column_span_column GtkComboBox
    @brief Sets the column with column span information for combo_box to be column_span.
    @param column_span A column in the model passed during construction

    The column span column contains integers which indicate how many columns an item should span.
 */
FALCON_FUNC ComboBox::set_column_span_column( VMARG )
{
    Item* i_span = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_span || i_span->isNil() || !i_span->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_combo_box_set_column_span_column( (GtkComboBox*)_obj, i_span->asInteger() );
}


/*#
    @method get_active GtkComboBox
    @brief Returns the index of the currently active item, or -1 if there's no active item.
    @return An integer which is the index of the currently active item, or -1 if there's no active item.

    If the model is a non-flat treemodel, and the active item is not an immediate child
    of the root of the tree, this function returns gtk_tree_path_get_indices (path)[0],
    where path is the GtkTreePath of the active item.
 */
FALCON_FUNC ComboBox::get_active( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_combo_box_get_active( (GtkComboBox*)_obj ) );
}


/*#
    @method set_active GtkComboBox
    @brief Sets the active item of combo_box to be the item at index.
    @param index An index in the model passed during construction, or -1 to have no active item
 */
FALCON_FUNC ComboBox::set_active( VMARG )
{
    Item* i_idx = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_idx || i_idx->isNil() || !i_idx->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_combo_box_set_active( (GtkComboBox*)_obj, i_idx->asInteger() );
}


//FALCON_FUNC ComboBox::get_active_iter( VMARG );

//FALCON_FUNC ComboBox::set_active_iter( VMARG );

//FALCON_FUNC ComboBox::get_model( VMARG );

//FALCON_FUNC ComboBox::set_model( VMARG );


/*#
    @method new_text GtkComboBox
    @brief Convenience function which constructs a new text combo box, which is a GtkComboBox just displaying strings.
    @return A new text combo box

    If you use this function to create a text combo box, you should only manipulate
    its data source with the following convenience functions:
    append_text(), insert_text(), prepend_text() and remove_text().
 */
FALCON_FUNC ComboBox::new_text( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    GtkWidget* wdt = gtk_combo_box_new_text();
    vm->retval( new Gtk::ComboBox(
        vm->findWKI( "GtkComboBox" )->asClass(), (GtkComboBox*) wdt ) );
}


/*#
    @method append_text GtkComboBox
    @brief Appends string to the list of strings stored in combo_box.
    @param text a string

    Note that you can only use this function with combo boxes constructed with
    gtk_combo_box_new_text().
 */
FALCON_FUNC ComboBox::append_text( VMARG )
{
    Gtk::ArgCheck<1> args( vm, "S" );

    const char* txt = args.getCString( 0 );

    MYSELF;
    GET_OBJ( self );
    gtk_combo_box_append_text( (GtkComboBox*)_obj, txt );
}


/*#
    @method insert_text GtkComboBox
    @brief Inserts string at position in the list of strings stored in combo_box.
    @param position An index to insert text
    @param text A string

    Note that you can only use this function with combo boxes constructed with
    gtk_combo_box_new_text().
 */
FALCON_FUNC ComboBox::insert_text( VMARG )
{
    Gtk::ArgCheck<1> args( vm, "I,S" );

    gint index = args.getInteger( 0 );
    const char* txt = args.getCString( 1 );

    MYSELF;
    GET_OBJ( self );
    gtk_combo_box_insert_text( (GtkComboBox*)_obj, index, txt );
}


/*#
    @method prepend_text GtkComboBox
    @brief Prepends string to the list of strings stored in combo_box.
    @param text A string

    Note that you can only use this function with combo boxes constructed with
    gtk_combo_box_new_text().
 */
FALCON_FUNC ComboBox::prepend_text( VMARG )
{
    Gtk::ArgCheck<1> args( vm, "S" );

    const char* txt = args.getCString( 0 );

    MYSELF;
    GET_OBJ( self );
    gtk_combo_box_prepend_text( (GtkComboBox*)_obj, txt );
}


/*#
    @method remove_text GtkComboBox
    @brief Removes the string at position from combo_box.
    @param position Index of the item to remove

    Note that you can only use this function with combo boxes constructed with
    gtk_combo_box_new_text().
 */
FALCON_FUNC ComboBox::remove_text( VMARG )
{
    Item* i_pos = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_pos || i_pos->isNil() || !i_pos->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_combo_box_remove_text( (GtkComboBox*)_obj, i_pos->asInteger() );
}


/*#
    @method get_active_text GtkComboBox
    @brief Returns the currently active string in combo_box or NULL if none is selected.
    @return the currently active text

    Note that you can only use this function with combo boxes constructed with
    gtk_combo_box_new_text() and with GtkComboBoxEntrys.
 */
FALCON_FUNC ComboBox::get_active_text( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gchar* txt = gtk_combo_box_get_active_text( (GtkComboBox*)_obj );
    if ( txt )
    {
        String* s = new String( txt );
        s->bufferize();
        vm->retval( s );
        g_free( txt );
    }
    else
        vm->retnil();
}


/*#
    @method popup GtkComboBox
    @brief Pops up the menu or dropdown list of combo_box.

    This function is mostly intended for use by accessibility technologies;
    applications should have little use for it.
 */
FALCON_FUNC ComboBox::popup( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_combo_box_popup( (GtkComboBox*)_obj );
}


/*#
    @method popdown GtkComboBox
    @brief Hides the menu or dropdown list of combo_box.

    This function is mostly intended for use by accessibility technologies;
    applications should have little use for it.
 */
FALCON_FUNC ComboBox::popdown( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_combo_box_popdown( (GtkComboBox*)_obj );
}


//FALCON_FUNC ComboBox::get_popup_accessible( VMARG );

//FALCON_FUNC ComboBox::get_row_separator_func( VMARG );

//FALCON_FUNC ComboBox::set_row_separator_func( VMARG );


/*#
    @method set_add_tearoffs GtkComboBox
    @brief Sets whether the popup menu should have a tearoff menu item.
    @param add_tearoffs true to add tearoff menu items
 */
FALCON_FUNC ComboBox::set_add_tearoffs( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_combo_box_set_add_tearoffs( (GtkComboBox*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method get_add_tearoffs GtkComboBox
    @brief Gets the current value of the :add-tearoffs property.
    @return the current value of the :add-tearoffs property.
 */
FALCON_FUNC ComboBox::get_add_tearoffs( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_combo_box_get_add_tearoffs( (GtkComboBox*)_obj ) );
}


/*#
    @method set_title GtkComboBox
    @brief Sets the menu's title in tearoff mode.
    @param title a title for the menu in tearoff mode
 */
FALCON_FUNC ComboBox::set_title( VMARG )
{
    Gtk::ArgCheck<1> args( vm, "S" );

    const char* title = args.getCString( 0 );

    MYSELF;
    GET_OBJ( self );
    gtk_combo_box_set_title( (GtkComboBox*)_obj, title );
}


/*#
    @method get_title GtkComboBox
    @brief Gets the current title of the menu in tearoff mode.
    @return the menu's title in tearoff mode.
 */
FALCON_FUNC ComboBox::get_title( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    const gchar* title = gtk_combo_box_get_title( (GtkComboBox*)_obj );
    if ( title )
        vm->retval( new String( title ) );
    else
        vm->retnil();
}


/*#
    @method set_focus_on_click GtkComboBox
    @brief Sets whether the combo box will grab focus when it is clicked with the mouse.
    @param focus_on_click whether the combo box grabs focus when clicked with the mouse

    Making mouse clicks not grab focus is useful in places like toolbars where you
    don't want the keyboard focus removed from the main area of the application.
 */
FALCON_FUNC ComboBox::set_focus_on_click( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_combo_box_set_focus_on_click( (GtkComboBox*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method get_focus_on_click GtkComboBox
    @brief Returns whether the combo box grabs focus when it is clicked with the mouse.
    @return TRUE if the combo box grabs focus when it is clicked with the mouse.
 */
FALCON_FUNC ComboBox::get_focus_on_click( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_combo_box_get_focus_on_click( (GtkComboBox*)_obj ) );
}


/*#
    @method set_button_sensitivity GtkComboBox
    @brief Sets whether the dropdown button of the combo box should be always sensitive (GTK_SENSITIVITY_ON), never sensitive (GTK_SENSITIVITY_OFF) or only if there is at least one item to display (GTK_SENSITIVITY_AUTO).
    @param sensitivity (GtkSensitivityType) specify the sensitivity of the dropdown button
 */
FALCON_FUNC ComboBox::set_button_sensitivity( VMARG )
{
    Item* i_sens = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_sens || i_sens->isNil() || !i_sens->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_combo_box_set_button_sensitivity( (GtkComboBox*)_obj,
            (GtkSensitivityType) i_sens->asInteger() );
}


/*#
    @method get_button_sensitivity GtkComboBox
    @brief Returns whether the combo box sets the dropdown button sensitive or not when there are no items in the model.
    @return GTK_SENSITIVITY_ON if the dropdown button is sensitive when the model is empty, GTK_SENSITIVITY_OFF  if the button is always insensitive or GTK_SENSITIVITY_AUTO if it is only sensitive as long as the model has one item to be selected.
 */
FALCON_FUNC ComboBox::get_button_sensitivity( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_combo_box_get_button_sensitivity( (GtkComboBox*)_obj ) );
}


} // Gtk
} // Falcon
