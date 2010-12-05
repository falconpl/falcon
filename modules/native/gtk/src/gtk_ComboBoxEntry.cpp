/**
 *  \file gtk_ComboBoxEntry.cpp
 */

#include "gtk_ComboBoxEntry.hpp"

#include "gtk_CellEditable.hpp"

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void ComboBoxEntry::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_ComboBoxEntry = mod->addClass( "GtkComboBoxEntry", &ComboBoxEntry::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkComboBox" ) );
    c_ComboBoxEntry->getClassDef()->addInheritance( in );

    Gtk::MethodTab methods[] =
    {
    //{ "new_with_model",     &ComboBoxEntry::new_with_model },
    { "new_text",           &ComboBoxEntry::new_text },
    { "set_text_column",    &ComboBoxEntry::set_text_column },
    { "get_text_column",    &ComboBoxEntry::get_text_column },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_ComboBoxEntry, meth->name, meth->cb );

    Gtk::CellEditable::clsInit( mod, c_ComboBoxEntry );
}


ComboBoxEntry::ComboBoxEntry( const Falcon::CoreClass* gen, const GtkComboBoxEntry* box )
    :
    Gtk::CoreGObject( gen, (GObject*) box )
{}


Falcon::CoreObject* ComboBoxEntry::factory( const Falcon::CoreClass* gen, void* box, bool )
{
    return new ComboBoxEntry( gen, (GtkComboBoxEntry*) box );
}


/*#
    @class GtkComboBoxEntry
    @brief A text entry field with a dropdown list

    A GtkComboBoxEntry is a widget that allows the user to choose from a list of
    valid choices or enter a different value. It is very similar to a GtkComboBox,
    but it displays the selected value in an entry to allow modifying it.

    In contrast to a GtkComboBox, the underlying model of a GtkComboBoxEntry must
    always have a text column (see gtk_combo_box_entry_set_text_column()), and the
    entry will show the content of the text column in the selected row. To get the
    text from the entry, use gtk_combo_box_get_active_text().

    The changed signal will be emitted while typing into a GtkComboBoxEntry, as well
    as when selecting an item from the GtkComboBoxEntry's list.
    Use gtk_combo_box_get_active() or gtk_combo_box_get_active_iter() to discover
    whether an item was actually selected from the list.

    Connect to the activate signal of the GtkEntry (use gtk_bin_get_child()) to
    detect when the user actually finishes entering text.

    The convenience API to construct simple text-only GtkComboBoxes can also be
    used with GtkComboBoxEntrys which have been constructed with gtk_combo_box_entry_new_text().

    If you have special needs that go beyond a simple entry (e.g. input validation),
    it is possible to replace the child entry by a different widget using
    gtk_container_remove() and gtk_container_add().
 */
FALCON_FUNC ComboBoxEntry::init( VMARG )
{
#ifndef NO_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GtkWidget* wdt = gtk_combo_box_entry_new();
    self->setObject( (GObject*) wdt );
}


//FALCON_FUNC ComboBoxEntry::new_with_model( VMARG );


/*#
    @method new_text GtkComboBoxEntry
    @brief Create a new text GtkComboBoxEntry.

    Convenience function which constructs a new editable text combo box, which is
    a GtkComboBoxEntry just displaying strings. If you use this function to create
    a text combo box, you should only manipulate its data source with the following
    convenience functions: append_text(), insert_text(), prepend_text() and remove_text().
 */
FALCON_FUNC ComboBoxEntry::new_text( VMARG )
{
#ifndef NO_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    GtkWidget* wdt = gtk_combo_box_entry_new_text();
    vm->retval( new Gtk::ComboBoxEntry(
        vm->findWKI( "GtkComboBoxEntry" )->asClass(), (GtkComboBoxEntry*) wdt ) );
}


/*#
    @method set_text_column GtkComboBoxEntry
    @brief Sets the model column which entry_box should use to get strings from to be text_column.
    @param text_column A column in model to get the strings from.
 */
FALCON_FUNC ComboBoxEntry::set_text_column( VMARG )
{
    Item* i_col = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_col || i_col->isNil() || !i_col->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_combo_box_entry_set_text_column( (GtkComboBoxEntry*)_obj,
            i_col->asInteger() );
}


/*#
    @method get_text_column GtkComboBoxEntry
    @brief Returns the column which entry_box is using to get the strings from.
    @return A column in the data source model of entry box.
 */
FALCON_FUNC ComboBoxEntry::get_text_column( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_combo_box_entry_get_text_column( (GtkComboBoxEntry*)_obj ) );
}


} // Gtk
} // Falcon
