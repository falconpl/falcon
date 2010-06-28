#ifndef GTK_ENTRY_HPP
#define GTK_ENTRY_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::Entry
 */
class Entry
    :
    public Gtk::CoreGObject
{
public:

    Entry( const Falcon::CoreClass*, const GtkEntry* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC signal_activate( VMARG );

    static void on_activate( GtkEntry*, gpointer );

    static FALCON_FUNC signal_backspace( VMARG );

    static void on_backspace( GtkEntry*, gpointer );

    static FALCON_FUNC signal_copy_clipboard( VMARG );

    static void on_copy_clipboard( GtkEntry*, gpointer );

    static FALCON_FUNC signal_cut_clipboard( VMARG );

    static void on_cut_clipboard( GtkEntry*, gpointer );

    static FALCON_FUNC signal_delete_from_cursor( VMARG );

    static void on_delete_from_cursor( GtkEntry*, GtkDeleteType, gint, gpointer );

#if 0 // todo: missing GdkEvent
#if GTK_CHECK_VERSION( 2, 16, 0 )
    static FALCON_FUNC signal_icon_press( VMARG );

    static void on_icon_press( GtkEntry*, GtkEntryIconPosition, GdkEvent*, gpointer );

    static FALCON_FUNC signal_icon_release( VMARG );

    static void on_icon_release( GtkEntry*, GtkEntryIconPosition, GdkEvent*, gpointer );
#endif
#endif

    static FALCON_FUNC signal_insert_at_cursor( VMARG );

    static void on_insert_at_cursor( GtkEntry*, gchar*, gpointer );

    static FALCON_FUNC signal_move_cursor( VMARG );

    static void on_move_cursor( GtkEntry*, GtkMovementStep, gint, gboolean, gpointer );

    static FALCON_FUNC signal_paste_clipboard( VMARG );

    static void on_paste_clipboard( GtkEntry*, gpointer );

    static FALCON_FUNC signal_populate_popup( VMARG );

    static void on_populate_popup( GtkEntry*, GtkMenu*, gpointer );

#if GTK_CHECK_VERSION( 2, 20, 0 )
    static FALCON_FUNC signal_preedit_changed( VMARG );

    static void on_preedit_changed( GtkEntry*, gchar*, gpointer );
#endif

    static FALCON_FUNC signal_toggle_overwrite( VMARG );

    static void on_toggle_overwrite( GtkEntry*, gpointer );

#if GTK_CHECK_VERSION( 2, 18, 0 )
    static FALCON_FUNC new_with_buffer( VMARG );
#endif

    static FALCON_FUNC new_with_max_length( VMARG );

#if GTK_CHECK_VERSION( 2, 18, 0 )
    static FALCON_FUNC get_buffer( VMARG );

    static FALCON_FUNC set_buffer( VMARG );
#endif

    static FALCON_FUNC set_text( VMARG );

    //static FALCON_FUNC append_text( VMARG );

    //static FALCON_FUNC prepend_text( VMARG );

    //static FALCON_FUNC set_position( VMARG );

    static FALCON_FUNC get_text( VMARG );

#if GTK_CHECK_VERSION( 2, 14, 0 )
    static FALCON_FUNC get_text_length( VMARG );
#endif

#if 0 // deprecated
    static FALCON_FUNC select_region( VMARG );
#endif

    static FALCON_FUNC set_visibility( VMARG );

    static FALCON_FUNC set_invisible_char( VMARG );

#if GTK_CHECK_VERSION( 2, 16, 0 )
    static FALCON_FUNC unset_invisible_char( VMARG );
#endif

#if 0
    static FALCON_FUNC set_editable( VMARG );
#endif

    static FALCON_FUNC set_max_length( VMARG );

    static FALCON_FUNC get_activates_default( VMARG );

    static FALCON_FUNC get_has_frame( VMARG );

    static FALCON_FUNC get_inner_border( VMARG );

    static FALCON_FUNC get_width_chars( VMARG );

    static FALCON_FUNC set_activates_default( VMARG );

    static FALCON_FUNC set_has_frame( VMARG );

    //static FALCON_FUNC set_inner_border( VMARG );

    static FALCON_FUNC set_width_chars( VMARG );

    static FALCON_FUNC get_invisible_char( VMARG );

    static FALCON_FUNC set_alignment( VMARG );

    static FALCON_FUNC get_alignment( VMARG );

#if GTK_CHECK_VERSION( 2, 14, 0 )
    static FALCON_FUNC set_overwrite_mode( VMARG );

    static FALCON_FUNC get_overwrite_mode( VMARG );
#endif

    //static FALCON_FUNC get_layout( VMARG );

    static FALCON_FUNC get_layout_offsets( VMARG );

    static FALCON_FUNC layout_index_to_text_index( VMARG );

    static FALCON_FUNC text_index_to_layout_index( VMARG );

    static FALCON_FUNC get_max_length( VMARG );

    static FALCON_FUNC get_visibility( VMARG );

    //static FALCON_FUNC set_completion( VMARG );

    //static FALCON_FUNC get_completion( VMARG );

    static FALCON_FUNC set_cursor_hadjustment( VMARG );

    static FALCON_FUNC get_cursor_hadjustment( VMARG );

#if GTK_CHECK_VERSION( 2, 16, 0 )
    static FALCON_FUNC set_progress_fraction( VMARG );

    static FALCON_FUNC get_progress_fraction( VMARG );

    static FALCON_FUNC set_progress_pulse_step( VMARG );

    static FALCON_FUNC get_progress_pulse_step( VMARG );

    static FALCON_FUNC progress_pulse( VMARG );
#endif

#if GTK_CHECK_VERSION( 2, 22, 0 )
    //static FALCON_FUNC im_context_filter_keypress( VMARG );
    //static FALCON_FUNC reset_im_context( VMARG );
#endif

#if GTK_CHECK_VERSION( 2, 16, 0 )
    static FALCON_FUNC set_icon_from_pixbuf( VMARG );

    static FALCON_FUNC set_icon_from_stock( VMARG );

    static FALCON_FUNC set_icon_from_icon_name( VMARG );

    //static FALCON_FUNC set_icon_from_gicon( VMARG );

    static FALCON_FUNC get_icon_storage_type( VMARG );

    static FALCON_FUNC get_icon_pixbuf( VMARG );

    static FALCON_FUNC get_icon_stock( VMARG );

    static FALCON_FUNC get_icon_name( VMARG );

    //static FALCON_FUNC get_icon_gicon( VMARG );

    static FALCON_FUNC set_icon_activatable( VMARG );

    static FALCON_FUNC get_icon_activatable( VMARG );

    static FALCON_FUNC set_icon_sensitive( VMARG );

    static FALCON_FUNC get_icon_sensitive( VMARG );

    static FALCON_FUNC get_icon_at_pos( VMARG );

    static FALCON_FUNC set_icon_tooltip_text( VMARG );

    static FALCON_FUNC get_icon_tooltip_text( VMARG );

    static FALCON_FUNC set_icon_tooltip_markup( VMARG );

    static FALCON_FUNC get_icon_tooltip_markup( VMARG );

    //static FALCON_FUNC set_icon_drag_source( VMARG );

    //static FALCON_FUNC get_current_icon_drag_source( VMARG );
#endif // GTK_CHECK_VERSION( 2, 16, 0 )

#if GTK_CHECK_VERSION( 2, 20, 0 )
    //static FALCON_FUNC get_icon_window( VMARG );

    //static FALCON_FUNC get_text_window( VMARG );
#endif

};


} // Gtk
} // Falcon

#endif // !GTK_ENTRY_HPP
