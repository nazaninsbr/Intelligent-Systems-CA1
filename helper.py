import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def read_mat_file(path, field_to_return):
	content = sio.loadmat(path)
	return content[field_to_return]

def set_height_labels_for_bar_plot(rects, ax):
	for rect in rects:
		height = rect.get_height()
		ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

def draw_bar_plot(x_label, y_label, x, y):
	fig, ax = plt.subplots()
	rects1 = ax.bar(x, y)
	ax.set_xlabel(x_label)
	ax.set_ylabel(y_label)
	set_height_labels_for_bar_plot(rects1, ax)
	plt.subplots_adjust(bottom=0.2, top=0.9)
	plt.show()

def draw_scatter_plot(x, y, color, x_label, y_label, title):
	fig, ax = plt.subplots()
	ax.scatter(x, y, c=color, alpha=0.8, edgecolors='none')

	ax.set_xlabel(x_label, fontsize=10)
	ax.set_ylabel(y_label, fontsize=10)
	ax.set_title('{}'.format(title))
	fig.tight_layout()

	plt.show()

def draw_simple_scatter_plot(x, y, x_label, y_label,title):
	fig, ax = plt.subplots()
	ax.scatter(x, y)
	fig.tight_layout()
	ax.set_xlabel(x_label, fontsize=10)
	ax.set_ylabel(y_label, fontsize=10)
	ax.set_title('{}'.format(title))

	plt.subplots_adjust(bottom=0.15, top=0.9)
	plt.show()

def draw_simple_connected_plot(x, y, x_label, y_label,title):
	fig, ax = plt.subplots()
	ax.plot(x, y)
	fig.tight_layout()
	ax.set_xlabel(x_label, fontsize=10)
	ax.set_ylabel(y_label, fontsize=10)
	ax.set_title('{}'.format(title))

	plt.subplots_adjust(bottom=0.15, top=0.9)
	plt.show()

def draw_scatter_and_line_plot(line, x, y, color, x_label, y_label, title):
	fig = plt.figure()

	gs1 = gridspec.GridSpec(2, 1)
	ax1 = fig.add_subplot(gs1[0])
	ax2 = fig.add_subplot(gs1[1])

	ax1.scatter(x, y, c=color, alpha=0.8, edgecolors='none')

	ax1.set_xlabel(x_label, fontsize=10)
	ax1.set_ylabel(y_label, fontsize=10)
	ax1.set_title('{}'.format(title))
	
	ax2.plot(line)

	gs1.tight_layout(fig)
	plt.show()



