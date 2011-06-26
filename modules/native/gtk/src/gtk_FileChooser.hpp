#ifndef GTK_FILECHOOSER_HPP
#define GTK_FILECHOOSER_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \namespace Falcon::Gtk::FileChooser
 */
namespace FileChooser {

void clsInit( Falcon::Module*, Falcon::Symbol* );

FALCON_FUNC signal_confirm_overwrite( VMARG );

GtkFileChooserConfirmation on_confirm_overwrite( GtkFileChooser*, gpointer );

FALCON_FUNC signal_current_folder_changed( VMARG );

void on_current_folder_changed( GtkFileChooser*, gpointer );

FALCON_FUNC signal_file_activated( VMARG );

void on_file_activated( GtkFileChooser*, gpointer );

FALCON_FUNC signal_selection_changed( VMARG );

void on_selection_changed( GtkFileChooser*, gpointer );

FALCON_FUNC signal_update_preview( VMARG );

void on_update_preview( GtkFileChooser*, gpointer );

FALCON_FUNC set_action( VMARG );

FALCON_FUNC get_action( VMARG );

FALCON_FUNC set_local_only( VMARG );

FALCON_FUNC get_local_only( VMARG );

FALCON_FUNC set_select_multiple( VMARG );

FALCON_FUNC get_select_multiple( VMARG );

FALCON_FUNC set_show_hidden( VMARG );

FALCON_FUNC get_show_hidden( VMARG );

FALCON_FUNC set_do_overwrite_confirmation( VMARG );

FALCON_FUNC get_do_overwrite_confirmation( VMARG );

#if GTK_CHECK_VERSION( 2, 18, 0 )
FALCON_FUNC set_create_folders( VMARG );

FALCON_FUNC get_create_folders( VMARG );
#endif

FALCON_FUNC set_current_name( VMARG );

FALCON_FUNC get_filename( VMARG );

FALCON_FUNC set_filename( VMARG );

FALCON_FUNC select_filename( VMARG );

FALCON_FUNC unselect_filename( VMARG );

FALCON_FUNC select_all( VMARG );

FALCON_FUNC unselect_all( VMARG );

FALCON_FUNC get_filenames( VMARG );

FALCON_FUNC set_current_folder( VMARG );

FALCON_FUNC get_current_folder( VMARG );

FALCON_FUNC get_uri( VMARG );

FALCON_FUNC set_uri( VMARG );

FALCON_FUNC select_uri( VMARG );

FALCON_FUNC unselect_uri( VMARG );

FALCON_FUNC get_uris( VMARG );

FALCON_FUNC set_current_folder_uri( VMARG );

FALCON_FUNC get_current_folder_uri( VMARG );

FALCON_FUNC set_preview_widget( VMARG );

FALCON_FUNC get_preview_widget( VMARG );

FALCON_FUNC set_preview_widget_active( VMARG );

FALCON_FUNC get_preview_widget_active( VMARG );

FALCON_FUNC set_use_preview_label( VMARG );

FALCON_FUNC get_use_preview_label( VMARG );

FALCON_FUNC get_preview_filename( VMARG );

FALCON_FUNC get_preview_uri( VMARG );

FALCON_FUNC set_extra_widget( VMARG );

FALCON_FUNC get_extra_widget( VMARG );

FALCON_FUNC add_filter( VMARG );

#if 0
FALCON_FUNC remove_filter( VMARG );

FALCON_FUNC list_filters( VMARG );

FALCON_FUNC set_filter( VMARG );

FALCON_FUNC get_filter( VMARG );

FALCON_FUNC add_shortcut_folder( VMARG );

FALCON_FUNC remove_shortcut_folder( VMARG );

FALCON_FUNC list_shortcut_folders( VMARG );

FALCON_FUNC add_shortcut_folder_uri( VMARG );

FALCON_FUNC remove_shortcut_folder_uri( VMARG );

FALCON_FUNC list_shortcut_folder_uris( VMARG );

#if GTK_CHECK_VERSION( 2, 14, 0 )
FALCON_FUNC get_current_folder_file( VMARG );

FALCON_FUNC get_file( VMARG );

FALCON_FUNC get_files( VMARG );

FALCON_FUNC get_preview_file( VMARG );

FALCON_FUNC select_file( VMARG );

FALCON_FUNC set_current_folder_file( VMARG );

FALCON_FUNC set_file( VMARG );

FALCON_FUNC unselect_file( VMARG );
#endif // GTK_CHECK_VERSION( 2, 14, 0 )
#endif

} // FileChooser
} // Gtk
} // Falcon

#endif // !GTK_FILECHOOSER_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
