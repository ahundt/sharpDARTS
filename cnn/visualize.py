import sys
import genotypes
from graphviz import Digraph


def plot(genotype, filename):
  g = Digraph(
      format='pdf',
      edge_attr=dict(fontsize='32', fontname="times"),
      node_attr=dict(style='filled', shape='rect', align='center', fontsize='32', height='0.7', width='0.7', penwidth='2', fontname="times"),
      engine='dot')
  g.body.extend(['rankdir=LR'])

  g.node(str(0), label="c_{k-2}", fillcolor='darkseagreen2')
  g.node(str(1), label="c_{k-1}", fillcolor='darkseagreen2')
  assert len(genotype) % 2 == 0
  steps = len(genotype) // 2

  for i in range(steps):
    g.node(str(i), label='+', fillcolor='lightblue')
    # g.node(str(i), label='+_{' + str(i) + '}', fillcolor='lightblue')

  for i in range(steps):
    for k in [2*i, 2*i + 1]:
      op, j = genotype[k]
      if j == 0:
        u = "c_{k-2}"
      elif j == 1:
        u = "c_{k-1}"
      else:
        u = str(j-2)
      v = str(i)
      name_ijk = str(i) + '_' + str(j) + '_' + str(k) + op
      g.node(name_ijk, label=op, fillcolor='darkseagreen2')
      g.edge(u, name_ijk, fillcolor="black")
      g.edge(name_ijk, v, fillcolor="black")

  g.node("c_{k}", fillcolor='palegoldenrod')
  for i in range(steps):
    g.edge(str(i), "c_{k}", fillcolor="black")

  g.render(filename, view=True)


if __name__ == '__main__':
  if len(sys.argv) != 2:
    print("usage:\n python {} ARCH_NAME".format(sys.argv[0]))
    sys.exit(1)

  genotype_name = sys.argv[1]
  try:
    genotype = eval('genotypes.{}'.format(genotype_name))
  except AttributeError:
    print("{} is not specified in genotypes.py".format(genotype_name))
    sys.exit(1)

  plot(genotype.normal, genotype_name + "_normal")
  plot(genotype.reduce, genotype_name + "_reduction")

