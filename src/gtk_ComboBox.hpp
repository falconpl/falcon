#ifndef GTK_COMBOBOX_HPP
#define GTK_COMBOBOX_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::ComboBox
 */
class ComboBox
    :
    public Gtk::CoreGObject
{
public:

    ComboBox( const Falcon::CoreClass*, const GtkComboBox* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC signal_changed( VMARG );

    static void on_changed( GtkComboBox*, gpointer );

    static FALCON_FUNC signal_move_active( VMARG );

    static void on_move_active( GtkComboBox*, GtkScrollType, gpointer );

    static FALCON_FUNC signal_popdown( VMARG );

    static void on_popdown( GtkComboBox*, gpointer );

    static FALCON_FUNC signal_popup( VMARG );

    static void on_popup( GtkComboBox*, gpointer );

    //static FALCON_FUNC new_with_model( VMARG );

    static FALCON_FUNC get_wrap_width( VMARG );

    static FALCON_FUNC set_wrap_width( VMARG );

    static FALCON_FUNC get_row_span_column( VMARG );

    static FALCON_FUNC set_row_span_column( VMARG );

    static FALCON_FUNC get_column_span_column( VMARG );

    static FALCON_FUNC set_column_span_column( VMARG );

    static FALCON_FUNC get_active( VMARG );

    static FALCON_FUNC set_active( VMARG );

    //static FALCON_FUNC get_active_iter( VMARG );

    //static FALCON_FUNC set_active_iter( VMARG );

    //static FALCON_FUNC get_model( VMARG );

    //static FALCON_FUNC set_model( VMARG );

    static FALCON_FUNC new_text( VMARG );

    static FALCON_FUNC append_text( VMARG );

    static FALCON_FUNC insert_text( VMARG );

    static FALCON_FUNC prepend_text( VMARG );

    static FALCON_FUNC remove_text( VMARG );

    static FALCON_FUNC get_active_text( VMARG );

    static FALCON_FUNC popup( VMARG );

    static FALCON_FUNC popdown( VMARG );

    //static FALCON_FUNC get_popup_accessible( VMARG );

    //static FALCON_FUNC get_row_separator_func( VMARG );

    //static FALCON_FUNC set_row_separator_func( VMARG );

    static FALCON_FUNC set_add_tearoffs( VMARG );

    static FALCON_FUNC get_add_tearoffs( VMARG );

    static FALCON_FUNC set_title( VMARG );

    static FALCON_FUNC get_title( VMARG );

    static FALCON_FUNC set_focus_on_click( VMARG );

    static FALCON_FUNC get_focus_on_click( VMARG );

    static FALCON_FUNC set_button_sensitivity( VMARG );

    static FALCON_FUNC get_button_sensitivity( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_COMBOBOX_HPP
